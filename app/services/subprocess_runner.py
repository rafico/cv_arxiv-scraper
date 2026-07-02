"""Run crash-prone native work in a child process.

faiss-cpu, torch, Pillow and pdfplumber are C extensions: a bug there can raise a
SIGSEGV or a glibc abort ("malloc(): unaligned tcache chunk detected"), which a
Python ``try/except`` cannot catch — it takes down the whole process. Running such
work in a short-lived child process turns a native crash into a catchable
``NativeCrashError`` in the parent, so the single web/scrape worker survives.

Isolation is on by default and can be disabled with ``CV_ARXIV_NATIVE_ISOLATION=0``
(the test suite does this so it can keep mocking the in-process functions).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from collections.abc import Callable
from queue import Empty
from typing import Any

LOGGER = logging.getLogger(__name__)

# "spawn" gives the child a fresh interpreter, so it loads its own copy of the
# native libraries instead of inheriting the parent's (possibly wedged) state.
_CTX = mp.get_context("spawn")

_ENV_FLAG = "CV_ARXIV_NATIVE_ISOLATION"


class NativeCrashError(RuntimeError):
    """The isolated child was killed by a signal (e.g. SIGSEGV / SIGABRT)."""


def isolation_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "1") != "0"


def _child(target: Callable[..., Any], args: tuple, kwargs: dict, queue) -> None:
    try:
        queue.put(("ok", target(*args, **kwargs)))
    except BaseException as exc:  # noqa: BLE001 - relay any failure to the parent
        try:
            queue.put(("err", exc))
        except Exception:  # noqa: BLE001 - exc may be unpicklable; relay a picklable surrogate
            queue.put(("err", RuntimeError(f"{type(exc).__name__}: {exc}")))


def run_isolated(target: Callable[..., Any], *args: Any, timeout: float | None = None, **kwargs: Any) -> Any:
    """Run ``target(*args, **kwargs)`` in a child process and return its result.

    Raises ``NativeCrashError`` if the child dies from a signal, ``TimeoutError`` if
    it overruns ``timeout``, or re-raises any Python exception the child raised. When
    isolation is disabled the target runs inline. ``target`` must be importable
    (module-level) and the arguments / return value must be picklable.
    """
    if not isolation_enabled():
        return target(*args, **kwargs)

    name = getattr(target, "__name__", repr(target))
    queue = _CTX.Queue()
    proc = _CTX.Process(target=_child, args=(target, args, kwargs, queue))
    proc.start()

    # Drain the queue BEFORE joining. A result larger than the OS pipe buffer
    # (~64KB) blocks the child on queue.put() until the parent reads it; joining
    # first would deadlock until the timeout and lose the output. Poll the queue
    # while the child is alive so a native crash (no result) is still detected
    # promptly instead of waiting the full timeout.
    result = None
    got_result = False
    start = time.monotonic()
    while True:
        try:
            remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - start))
            result = queue.get(timeout=0.5 if remaining is None else min(0.5, remaining))
            got_result = True
            break
        except Empty:
            if not proc.is_alive():
                break  # child exited without producing a result (likely a crash)
            if timeout is not None and time.monotonic() - start >= timeout:
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    # A child wedged in an uninterruptible native call can ignore
                    # SIGTERM; escalate to SIGKILL so join() cannot block forever.
                    proc.kill()
                    proc.join(timeout=5)
                raise TimeoutError(f"isolated call to {name} timed out after {timeout}s") from None

    # The child has put its result (or died); joining now is safe and lets its
    # feeder thread flush the pipe.
    proc.join()

    if not got_result:
        # A result can still be buffered in the pipe if the child produced it and
        # exited between two polls (proc.is_alive() went False before the feeder
        # thread flushed). Try one final non-blocking read before assuming a crash.
        try:
            result = queue.get_nowait()
            got_result = True
        except Empty:
            pass

    if not got_result:
        if proc.exitcode is not None and proc.exitcode < 0:
            raise NativeCrashError(f"{name} crashed in an isolated process (signal {-proc.exitcode})")
        raise NativeCrashError(f"{name} produced no result (exit code {proc.exitcode})")

    status, payload = result
    if status == "err":
        raise payload
    return payload
