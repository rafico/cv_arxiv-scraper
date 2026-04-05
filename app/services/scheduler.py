"""Built-in scrape scheduler — replaces external cron."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone

LOGGER = logging.getLogger(__name__)


class ScrapeScheduler:
    """Runs scrapes on a configurable daily schedule using a daemon thread."""

    def __init__(self) -> None:
        self._timer: threading.Timer | None = None
        self._enabled = False
        self._app = None
        self._daily_at: str = "08:00"
        self._lock = threading.Lock()

    def start(self, app, *, daily_at: str = "08:00") -> None:
        with self._lock:
            self._app = app
            self._daily_at = daily_at
            self._enabled = True
            self._schedule_next()

    def stop(self) -> None:
        with self._lock:
            self._enabled = False
            if self._timer:
                self._timer.cancel()
                self._timer = None

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


SCRAPE_SCHEDULER = ScrapeScheduler()
