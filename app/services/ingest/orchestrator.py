"""Coordinate ingest backends for scrape and backfill workflows."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import date

import requests

from app.services.ingest.arxiv_api_backend import ArxivApiBackend
from app.services.ingest.base import IngestMode, PaperCandidate
from app.services.ingest.rss_backend import RssFeedBackend
from app.services.text import utc_today

LOGGER = logging.getLogger(__name__)

RssCandidateFetcher = Callable[..., list[PaperCandidate]]
RollingWindowFetcher = Callable[..., list[PaperCandidate]]


class IngestOrchestrator:
    def __init__(
        self,
        *,
        rss_candidate_fetcher: RssCandidateFetcher | None = None,
        rolling_window_fetcher: RollingWindowFetcher | None = None,
        arxiv_api_backend: ArxivApiBackend | None = None,
    ):
        self._rss_candidate_fetcher = rss_candidate_fetcher or self._default_rss_candidate_fetcher
        self._rolling_window_fetcher = rolling_window_fetcher or self._default_rolling_window_fetcher
        self._arxiv_api_backend = arxiv_api_backend or ArxivApiBackend()

    def fetch(
        self,
        *,
        mode: IngestMode,
        session: requests.Session | None = None,
        feed_urls: Sequence[str] | None = None,
        rolling_window_days: int = 0,
        categories: Sequence[str] | None = None,
        start_dt: date | None = None,
        end_dt: date | None = None,
        max_results: int = 2000,
    ) -> list[PaperCandidate]:
        if mode == IngestMode.DAILY_WATCH:
            return self._fetch_daily_watch(
                feed_urls=feed_urls or [],
                rolling_window_days=rolling_window_days,
                session=session,
            )
        if mode == IngestMode.BACKFILL:
            if start_dt is None or end_dt is None:
                raise ValueError("BACKFILL mode requires start_dt and end_dt")
            return self._fetch_arxiv_api(
                categories=categories or [],
                start_dt=start_dt,
                end_dt=end_dt,
                max_results=max_results,
                session=session,
            )
        if mode == IngestMode.CATCH_UP:
            if start_dt is None:
                raise ValueError("CATCH_UP mode requires start_dt")
            return self._fetch_arxiv_api(
                categories=categories or [],
                start_dt=start_dt,
                end_dt=end_dt or utc_today(),
                max_results=max_results,
                session=session,
            )
        raise ValueError(f"Unsupported ingest mode: {mode}")

    def _fetch_daily_watch(
        self,
        *,
        feed_urls: Sequence[str],
        rolling_window_days: int,
        session: requests.Session | None = None,
    ) -> list[PaperCandidate]:
        normalized_feed_urls = [feed_url for feed_url in feed_urls if feed_url]
        rss_candidates = self._fetch_rss_candidates(normalized_feed_urls, session=session)

        if rolling_window_days <= 0:
            return rss_candidates

        try:
            rolling_candidates: list[PaperCandidate] = []
            for feed_url in normalized_feed_urls:
                rolling_candidates.extend(
                    self._rolling_window_fetcher(
                        rolling_window_days,
                        feed_url,
                        session=session,
                    )
                )
        except Exception as exc:
            LOGGER.warning("Rolling-window fetch failed: %s", exc)
            rolling_candidates = []

        return self._merge_candidates(rss_candidates, rolling_candidates)

    def _fetch_rss_candidates(
        self,
        feed_urls: Sequence[str],
        *,
        session: requests.Session | None = None,
    ) -> list[PaperCandidate]:
        candidates: list[PaperCandidate] = []
        feed_errors: list[Exception] = []

        for feed_url in feed_urls:
            try:
                candidates.extend(self._rss_candidate_fetcher(feed_url, session=session))
            except Exception as exc:
                LOGGER.warning("Failed to parse feed %s: %s", feed_url, exc)
                feed_errors.append(exc)

        if feed_errors and not candidates and len(feed_errors) == len(feed_urls):
            raise feed_errors[0]

        return candidates

    def _fetch_arxiv_api(
        self,
        *,
        categories: Sequence[str],
        start_dt: date,
        end_dt: date,
        max_results: int,
        session: requests.Session | None = None,
    ) -> list[PaperCandidate]:
        return self._arxiv_api_backend.fetch(
            categories=categories,
            start_dt=start_dt,
            end_dt=end_dt,
            max_results=max_results,
            session=session,
        )

    @staticmethod
    def _merge_candidates(primary: Sequence[PaperCandidate], secondary: Sequence[PaperCandidate]) -> list[PaperCandidate]:
        merged_candidates: dict[str, PaperCandidate] = {}
        for candidate in primary:
            merged_candidates[candidate.arxiv_id or candidate.link] = candidate
        for candidate in secondary:
            merged_candidates.setdefault(candidate.arxiv_id or candidate.link, candidate)
        return list(merged_candidates.values())

    @staticmethod
    def _default_rss_candidate_fetcher(
        feed_url: str,
        *,
        session: requests.Session | None = None,
    ) -> list[PaperCandidate]:
        return RssFeedBackend([feed_url]).fetch(session=session)

    @staticmethod
    def _default_rolling_window_fetcher(
        days: int,
        feed_url: str,
        *,
        session: requests.Session | None = None,
    ) -> list[PaperCandidate]:
        from app.services.enrichment import fetch_recent_papers

        return [
            PaperCandidate.from_entry_dict(entry)
            for entry in fetch_recent_papers(days, feed_url, session=session)
        ]
