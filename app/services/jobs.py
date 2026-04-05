"""Background scrape job orchestration and SSE streaming."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

from app.services.scrape_engine import execute_scrape
from app.services.text import now_utc

LOGGER = logging.getLogger(__name__)


@dataclass
class ScrapeJob:
    id: str
    started_at: datetime
    status: str = "running"
    events: list[tuple[str, dict]] = field(default_factory=list)
    finished_at: datetime | None = None
    condition: threading.Condition = field(default_factory=threading.Condition)


class ScrapeJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scrape-job")
        self._jobs: dict[str, ScrapeJob] = {}
        self._active_job_id: str | None = None

    def _trim_history(self, keep: int = 4, stale_hours: int = 2) -> None:
        if len(self._jobs) <= keep:
            return

        now = now_utc()
        completed_jobs = sorted(
            (job for job in self._jobs.values() if job.finished_at is not None),
            key=lambda job: job.finished_at or job.started_at,
        )
        for job in completed_jobs[:-keep]:
            self._jobs.pop(job.id, None)

        # Also evict jobs stuck in "running" state beyond the stale threshold.
        for job_id, job in list(self._jobs.items()):
            if (
                job.finished_at is None
                and job_id != self._active_job_id
                and (now - job.started_at).total_seconds() > stale_hours * 3600
            ):
                self._jobs.pop(job_id, None)

    def _publish(self, job_id: str, event: str, data: dict) -> None:
        terminal_events = {"done", "scrape_error", "skipped"}
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if event in terminal_events and self._active_job_id == job_id:
                self._active_job_id = None

        with job.condition:
            job.events.append((event, data))
            if event == "done":
                job.status = "finished"
                job.finished_at = now_utc()
            elif event == "scrape_error":
                job.status = "error"
                job.finished_at = now_utc()
            elif event == "skipped":
                job.status = "skipped"
                job.finished_at = now_utc()
            job.condition.notify_all()

    def _run_job(self, app, job_id: str, force: bool = False) -> None:
        try:
            execute_scrape(
                app,
                event_callback=lambda event, data: self._publish(job_id, event, data),
                force=force,
            )
        except Exception as exc:
            LOGGER.exception("Background scrape job failed")
            self._publish(job_id, "scrape_error", {"message": str(exc)})
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
                self._trim_history()

    def start_or_get_active(self, app, force: bool = False) -> ScrapeJob:
        with self._lock:
            if self._active_job_id and self._active_job_id in self._jobs:
                active_job = self._jobs[self._active_job_id]
                if active_job.finished_at is None and active_job.status == "running":
                    return active_job
                self._active_job_id = None

            job_id = uuid.uuid4().hex
            job = ScrapeJob(id=job_id, started_at=now_utc())
            self._jobs[job_id] = job
            self._active_job_id = job_id
            self._executor.submit(self._run_job, app, job_id, force)
            return job

    def get_status_snapshot(self) -> dict:
        """Thread-safe status snapshot for the polling endpoint."""
        with self._lock:
            if self._active_job_id and self._active_job_id in self._jobs:
                job = self._jobs[self._active_job_id]
                # A job can be marked finished/error before _active_job_id is cleared
                # in the worker's finally block; treat it as terminal immediately.
                if job.finished_at is not None or job.status in {"finished", "error"}:
                    return {
                        "running": False,
                        "terminal_status": job.status,
                        "job_id": job.id,
                    }
                return {"running": True, "status": job.status}

            # Check the most recently finished job for terminal state.
            completed = [j for j in self._jobs.values() if j.finished_at is not None]
            if completed:
                latest = max(completed, key=lambda j: j.finished_at)
                return {
                    "running": False,
                    "terminal_status": latest.status,
                    "job_id": latest.id,
                }

            return {"running": False}

    def stream_events(self, job_id: str, *, heartbeat_seconds: int = 15) -> Iterator[tuple[str, dict]]:
        job = self._jobs.get(job_id)
        if not job:
            yield "scrape_error", {"message": "Job not found"}
            return

        cursor = 0
        while True:
            heartbeat_event: tuple[str, dict] | None = None
            next_event: tuple[str, dict] | None = None
            with job.condition:
                if cursor < len(job.events):
                    next_event = job.events[cursor]
                    cursor += 1
                else:
                    if job.finished_at is not None:
                        break
                    job.condition.wait(timeout=heartbeat_seconds)
                    if cursor >= len(job.events) and job.finished_at is None:
                        heartbeat_event = (
                            "status",
                            {"phase": "heartbeat", "message": "Scrape still running..."},
                        )
                    elif cursor < len(job.events):
                        next_event = job.events[cursor]
                        cursor += 1
                    else:
                        continue

            if heartbeat_event is not None:
                yield heartbeat_event
                continue

            if next_event is None:
                raise RuntimeError("Expected event but got None")
            event, data = next_event
            yield next_event
            if event in {"done", "scrape_error", "skipped"} and cursor >= len(job.events):
                break

    def stream_for_request(self, app, force: bool = False):
        job = self.start_or_get_active(app, force=force)
        for event, data in self.stream_events(job.id):
            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"


SCRAPE_JOB_MANAGER = ScrapeJobManager()
