"""Tests for the native-work subprocess isolation helper."""

from __future__ import annotations

import os
import signal
import time

import pytest

from app.services.subprocess_runner import NativeCrashError, run_isolated

# Targets must be importable (module-level) so the spawned child can unpickle them.


def _double(value: int) -> int:
    return value * 2


def _raise_value_error() -> None:
    raise ValueError("boom from child")


def _self_terminate_with_signal() -> None:
    # Simulate a native crash (a real SIGSEGV/abort can't be caught in-process).
    os.kill(os.getpid(), signal.SIGKILL)


def _large_payload(size: int) -> bytes:
    return b"x" * size


def _sleep(seconds: float) -> int:
    time.sleep(seconds)
    return 1


@pytest.fixture
def _isolated(monkeypatch):
    monkeypatch.setenv("CV_ARXIV_NATIVE_ISOLATION", "1")


def test_returns_child_result(_isolated):
    assert run_isolated(_double, 21) == 42


def test_reraises_child_python_exception(_isolated):
    with pytest.raises(ValueError, match="boom from child"):
        run_isolated(_raise_value_error)


def test_signal_death_becomes_native_crash_error(_isolated):
    # A child killed by a signal must surface as a catchable error, not crash the parent.
    with pytest.raises(NativeCrashError):
        run_isolated(_self_terminate_with_signal)


def test_large_payload_returned_intact(_isolated):
    # A result larger than the OS pipe buffer (~64KB) used to deadlock: the child
    # blocked on queue.put() while the parent blocked on join(). Draining before
    # joining must return the full payload (here 1 MiB) without a TimeoutError.
    size = 1024 * 1024
    result = run_isolated(_large_payload, size, timeout=30)
    assert result == b"x" * size


def test_timeout_raises_and_kills_child(_isolated):
    with pytest.raises(TimeoutError):
        run_isolated(_sleep, 30, timeout=0.5)


def test_runs_inline_when_isolation_disabled(monkeypatch):
    monkeypatch.setenv("CV_ARXIV_NATIVE_ISOLATION", "0")
    # Inline mode returns directly and never spawns a process.
    assert run_isolated(_double, 5) == 10
