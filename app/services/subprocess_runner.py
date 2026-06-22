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
from collections.abc import Callable
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
        queue.put(("err", exc))


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
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(f"isolated call to {name} timed out after {timeout}s")

    if proc.exitcode is not None and proc.exitcode < 0:
        raise NativeCrashError(f"{name} crashed in an isolated process (signal {-proc.exitcode})")

    try:
        status, payload = queue.get_nowait()
    except Exception as exc:  # queue empty => child exited without producing a result
        raise NativeCrashError(f"{name} produced no result (exit code {proc.exitcode})") from exc

    if status == "err":
        raise payload
    return payload
