"""Off-request-thread thumbnail/teaser generation.

The dashboard renders one ``<img>`` per paper pointing at the thumbnail/teaser
routes. When the PNG is missing those routes must NOT generate it inline: a single
generation downloads a PDF and renders it in a subprocess and can hold a worker
thread for minutes, and with only 1-2 ``gthread`` worker threads (see
``run.py``) that freezes the whole UI. Instead the route enqueues the work here
and returns immediately; the cached file is served on a later request once warmed.

This mirrors the in-process singleton pattern of
``app.services.jobs.ScrapeJobManager`` but uses its own small pool so warming
never serializes behind a scrape.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.services.thumbnail_generator import generate_thumbnail

LOGGER = logging.getLogger(__name__)


class ThumbnailWarmer:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="thumb-warm")
        self._lock = threading.Lock()
        # Storage keys with a generation already scheduled/running, so a burst of
        # lazy <img> requests for the same paper only triggers one download+render.
        self._in_flight: set[str] = set()

    def warm(self, storage_key: str, pdf_link: str, static_dir: str | Path) -> None:
        """Schedule thumbnail+teaser generation for ``storage_key`` (no-op if one
        is already pending). Returns immediately; never blocks the caller."""
        if not storage_key or not pdf_link:
            return
        with self._lock:
            if storage_key in self._in_flight:
                return
            self._in_flight.add(storage_key)
        try:
            self._executor.submit(self._run, storage_key, pdf_link, str(static_dir))
        except Exception:
            # submit() raises (e.g. RuntimeError after the executor is shut down) before
            # _run — which clears _in_flight — is ever scheduled. Discard the key here so
            # this paper isn't permanently blocked from a future warm.
            with self._lock:
                self._in_flight.discard(storage_key)
            LOGGER.warning("Failed to enqueue thumbnail warm for %s", storage_key, exc_info=True)

    def _run(self, storage_key: str, pdf_link: str, static_dir: str) -> None:
        try:
            generate_thumbnail(storage_key, pdf_link, static_dir)
        except Exception:  # pragma: no cover - generate_thumbnail already guards itself
            LOGGER.warning("Background thumbnail warm failed for %s", storage_key, exc_info=True)
        finally:
            with self._lock:
                self._in_flight.discard(storage_key)


# Process-wide singleton. Like the scrape job manager, state lives in this
# process's memory only — fine because the app runs a single web worker.
THUMBNAIL_WARMER = ThumbnailWarmer()
