"""Built-in scrape scheduler — replaces external cron."""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_SCHEDULER_LOCK_FILENAME = ".scrape_scheduler.lock"


class ScrapeScheduler:
    """Runs scrapes on a configurable daily schedule using a daemon thread."""

    def __init__(self) -> None:
        self._timer: threading.Timer | None = None
        self._enabled = False
        self._app = None
        self._daily_at: str = "08:00"
        self._lock = threading.Lock()
        self._leader_lock_fd: int | None = None
        self._leader_lock_path: Path | None = None

    def start(self, app, *, daily_at: str = "08:00") -> None:
        with self._lock:
            desired_lock_path = Path(app.instance_path) / _SCHEDULER_LOCK_FILENAME
            if self._leader_lock_path != desired_lock_path:
                self._release_leader_lock()

            if self._leader_lock_fd is None and not self._acquire_leader_lock(desired_lock_path):
                self._enabled = False
                self._app = None
                return

            self._app = app
            self._daily_at = daily_at
            self._enabled = True
            if self._timer:
                self._timer.cancel()
            self._schedule_next()

    def stop(self) -> None:
        with self._lock:
            self._enabled = False
            self._app = None
            if self._timer:
                self._timer.cancel()
                self._timer = None
            self._release_leader_lock()

    def _seconds_until(self, time_str: str) -> float:
        now = datetime.now(timezone.utc)
        try:
            hour, minute = (int(x) for x in time_str.split(":"))
        except (ValueError, TypeError):
            LOGGER.warning("Invalid daily_at value %r, defaulting to 08:00", time_str)
            hour, minute = 8, 0
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return (target - now).total_seconds()

    def _schedule_next(self) -> None:
        if not self._enabled:
            return
        delay = self._seconds_until(self._daily_at)
        LOGGER.info("Next scheduled scrape in %.0f seconds (at %s UTC)", delay, self._daily_at)
        self._timer = threading.Timer(delay, self._run)
        self._timer.daemon = True
        self._timer.start()

    def _run(self) -> None:
        if not self._enabled or not self._app:
            return
        LOGGER.info("Scheduled scrape starting")
        try:
            from app.services.scrape_engine import execute_scrape

            execute_scrape(self._app)
        except Exception:
            LOGGER.exception("Scheduled scrape failed")
        finally:
            with self._lock:
                self._schedule_next()

    @property
    def next_run_at(self) -> str | None:
        if not self._enabled:
            return None
        now = datetime.now(timezone.utc)
        try:
            hour, minute = (int(x) for x in self._daily_at.split(":"))
        except (ValueError, TypeError):
            hour, minute = 8, 0
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target.strftime("%Y-%m-%d %H:%M UTC")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def schedule_time(self) -> str:
        return self._daily_at

    def _acquire_leader_lock(self, lock_path: Path) -> bool:
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(2):
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if attempt == 0 and self._clear_stale_lock(lock_path):
                    continue

                LOGGER.info("Scheduler lock already held for %s; skipping local scheduler start", lock_path)
                return False

            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
            self._leader_lock_fd = fd
            self._leader_lock_path = lock_path
            return True

        return False

    def _clear_stale_lock(self, lock_path: Path) -> bool:
        pid = self._read_lock_pid(lock_path)
        if pid is None:
            return False

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            try:
                lock_path.unlink()
                LOGGER.warning("Removed stale scheduler lock for dead pid %s", pid)
                return True
            except FileNotFoundError:
                return True
            except OSError:
                return False
        except PermissionError:
            return False

        return False

    @staticmethod
    def _read_lock_pid(lock_path: Path) -> int | None:
        try:
            raw_pid = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None

        try:
            return int(raw_pid)
        except ValueError:
            return None

    def _release_leader_lock(self) -> None:
        lock_path = self._leader_lock_path
        lock_fd = self._leader_lock_fd
        self._leader_lock_path = None
        self._leader_lock_fd = None

        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError:
                pass

        if lock_path is not None:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                LOGGER.warning("Failed to remove scheduler lock file %s", lock_path)


SCRAPE_SCHEDULER = ScrapeScheduler()
