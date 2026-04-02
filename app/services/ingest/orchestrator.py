"""Coordinate ingest backends for scrape and backfill workflows."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time

import requests

from app.services.ingest.arxiv_api_backend import ArxivApiBackend
from app.services.ingest.base import IngestMode, PaperCandidate
from app.services.ingest.rss_backend import RssFeedBackend
from app.services.text import now_utc, utc_today

LOGGER = logging.getLogger(__name__)

RssCandidateFetcher = Callable[..., list[PaperCandidate]]
RollingWindowFetcher = Callable[..., list[PaperCandidate]]
BackendRegistry = Mapping[str, type]
SyncStateReader = Callable[[Sequence[str]], Mapping[str, datetime | None]]
SyncStateWriter = Callable[..., None]
Clock = Callable[[], datetime]

BACKEND_REGISTRY: dict[str, type] = {
    "rss": RssFeedBackend,
    "arxiv_api": ArxivApiBackend,
}


class IngestOrchestrator:
    def __init__(
        self,
        *,
        rss_candidate_fetcher: RssCandidateFetcher | None = None,
        rolling_window_fetcher: RollingWindowFetcher | None = None,
        arxiv_api_backend: ArxivApiBackend | None = None,
        backend_registry: BackendRegistry | None = None,
        sync_state_reader: SyncStateReader | None = None,
        sync_state_writer: SyncStateWriter | None = None,
        clock: Clock | None = None,
    ):
        self._rss_candidate_fetcher = rss_candidate_fetcher or self._default_rss_candidate_fetcher
        self._rolling_window_fetcher = rolling_window_fetcher or self._default_rolling_window_fetcher
        self._arxiv_api_backend = arxiv_api_backend or ArxivApiBackend()
        self._backend_registry = dict(backend_registry or BACKEND_REGISTRY)
        self._sync_state_reader = sync_state_reader or self._default_sync_state_reader
        self._sync_state_writer = sync_state_writer or self._default_sync_state_writer
        self._clock = clock or now_utc

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
        backend_names: Sequence[str] | None = None,
    ) -> list[PaperCandidate]:
        selected_backend_names = self._resolve_backend_names(backend_names)
        if mode == IngestMode.DAILY_WATCH:
            return self._fetch_daily_watch(
                feed_urls=feed_urls or [],
                rolling_window_days=rolling_window_days,
                session=session,
                backend_names=selected_backend_names,
                explicit_selection=backend_names is not None,
            )
        if mode == IngestMode.BACKFILL:
            if start_dt is None or end_dt is None:
                raise ValueError("BACKFILL mode requires start_dt and end_dt")
            self._require_backend(selected_backend_names, backend_name="arxiv_api", mode=mode)
            return self._fetch_arxiv_api(
                categories=categories or [],
                start_dt=start_dt,
                end_dt=end_dt,
                max_results=max_results,
                session=session,
            )
        if mode == IngestMode.CATCH_UP:
            self._require_backend(selected_backend_names, backend_name="arxiv_api", mode=mode)
            return self._fetch_catch_up(
                categories=categories or [],
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
        backend_names: Sequence[str],
        explicit_selection: bool,
    ) -> list[PaperCandidate]:
        normalized_feed_urls = [feed_url for feed_url in feed_urls if feed_url]
        api_days = self._recent_fetch_days(
            rolling_window_days=rolling_window_days,
            explicit_selection=explicit_selection,
            backend_names=backend_names,
        )
        arxiv_api_active = "arxiv_api" in backend_names and api_days > 0

        rss_candidates: list[PaperCandidate] = []
        if "rss" in backend_names:
            rss_candidates = self._fetch_rss_candidates(
                normalized_feed_urls,
                session=session,
                raise_on_total_failure=not arxiv_api_active,
            )

        if api_days <= 0:
            return rss_candidates

        if "arxiv_api" not in backend_names:
            return rss_candidates

        rolling_candidates = self._fetch_recent_candidates(
            days=api_days,
            feed_urls=normalized_feed_urls,
            session=session,
            strict=not rss_candidates,
        )
        return self._merge_candidates(rss_candidates, rolling_candidates)

    def _fetch_recent_candidates(
        self,
        *,
        days: int,
        feed_urls: Sequence[str],
        session: requests.Session | None = None,
        strict: bool = False,
    ) -> list[PaperCandidate]:
        try:
            candidates: list[PaperCandidate] = []
            for feed_url in feed_urls:
                candidates.extend(
                    self._rolling_window_fetcher(
                        days,
                        feed_url,
                        session=session,
                    )
                )
            return candidates
        except Exception as exc:
            LOGGER.warning("Rolling-window fetch failed: %s", exc)
            if strict:
                raise
            return []

    def _fetch_rss_candidates(
        self,
        feed_urls: Sequence[str],
        *,
        session: requests.Session | None = None,
        raise_on_total_failure: bool = True,
    ) -> list[PaperCandidate]:
        candidates: list[PaperCandidate] = []
        feed_errors: list[Exception] = []

        for feed_url in feed_urls:
            try:
                candidates.extend(self._rss_candidate_fetcher(feed_url, session=session))
            except Exception as exc:
                LOGGER.warning("Failed to parse feed %s: %s", feed_url, exc)
                feed_errors.append(exc)

        if raise_on_total_failure and feed_errors and not candidates and len(feed_errors) == len(feed_urls):
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

    def _fetch_catch_up(
        self,
        *,
        categories: Sequence[str],
        end_dt: date,
        max_results: int,
        session: requests.Session | None = None,
    ) -> list[PaperCandidate]:
        if not categories:
            return []

        sync_state_by_category = self._sync_state_reader(categories)
        missing_categories = [category for category in categories if sync_state_by_category.get(category) is None]
        if missing_categories:
            raise ValueError(
                "CATCH_UP mode requires SyncState.last_synced_submitted_at for categories: "
                f"{', '.join(missing_categories)}"
            )

        synced_at = self._clock()
        synced_through = datetime.combine(end_dt, time.max) if end_dt != utc_today() else synced_at

        candidates: list[PaperCandidate] = []
        for category in categories:
            last_synced_at = sync_state_by_category.get(category)
            category_candidates = self._fetch_arxiv_api(
                categories=[category],
                start_dt=last_synced_at.date(),
                end_dt=end_dt,
                max_results=max_results,
                session=session,
            )
            candidates.extend(category_candidates)
            self._sync_state_writer(
                category,
                synced_through=synced_through,
                updated_at=synced_at,
                paper_count=len(category_candidates),
            )

        return candidates

    @staticmethod
    def _merge_candidates(primary: Sequence[PaperCandidate], secondary: Sequence[PaperCandidate]) -> list[PaperCandidate]:
        merged_candidates: dict[str, PaperCandidate] = {}
        for candidate in primary:
            merged_candidates[candidate.arxiv_id or candidate.link] = candidate
        for candidate in secondary:
            merged_candidates.setdefault(candidate.arxiv_id or candidate.link, candidate)
        return list(merged_candidates.values())

    def _resolve_backend_names(self, backend_names: Sequence[str] | None) -> list[str]:
        selected = list(backend_names) if backend_names is not None else list(self._backend_registry)
        unknown = [name for name in selected if name not in self._backend_registry]
        if unknown:
            raise ValueError(f"Unknown ingest backends: {', '.join(unknown)}")
        return selected

    @staticmethod
    def _default_sync_state_reader(categories: Sequence[str]) -> dict[str, datetime | None]:
        from app.models import SyncState

        states = SyncState.query.filter(SyncState.category.in_(list(categories))).all()
        return {state.category: state.last_synced_submitted_at for state in states}

    @staticmethod
    def _default_sync_state_writer(
        category: str,
        *,
        synced_through: datetime,
        updated_at: datetime,
        paper_count: int,
    ) -> None:
        from app.models import SyncState, db

        state = SyncState.query.filter_by(category=category).one_or_none()
        if state is None:
            state = SyncState(category=category)
            db.session.add(state)

        state.last_synced_submitted_at = synced_through
        state.last_synced_updated_at = updated_at
        state.last_synced_paper_count = paper_count
        db.session.commit()

    @staticmethod
    def _require_backend(
        backend_names: Sequence[str],
        *,
        backend_name: str,
        mode: IngestMode,
    ) -> None:
        if backend_name not in backend_names:
            raise ValueError(f"{mode.value.upper()} mode requires ingest backend '{backend_name}'")

    @staticmethod
    def _recent_fetch_days(
        *,
        rolling_window_days: int,
        explicit_selection: bool,
        backend_names: Sequence[str],
    ) -> int:
        if "arxiv_api" not in backend_names:
            return 0
        if rolling_window_days > 0:
            return rolling_window_days
        if explicit_selection:
            return 1
        return 0

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
