from __future__ import annotations

from datetime import date, datetime
from unittest import TestCase

from app.services.ingest import IngestMode, IngestOrchestrator, PaperCandidate


def _candidate(arxiv_id: str, title: str) -> PaperCandidate:
    return PaperCandidate(
        arxiv_id=arxiv_id,
        link=f"https://arxiv.org/abs/{arxiv_id}",
        title=title,
        author="Author A",
        authors_list=["Author A"],
    )


class IngestOrchestratorTests(TestCase):
    def test_daily_watch_merges_by_arxiv_id_with_rss_precedence(self):
        orchestrator = IngestOrchestrator(
            rss_candidate_fetcher=lambda feed_url, *, session=None: [_candidate("0001", f"rss:{feed_url}")],
            rolling_window_fetcher=lambda days, feed_url, *, session=None: [
                _candidate("0001", f"recent-dup:{feed_url}"),
                _candidate("0002", f"recent-new:{feed_url}"),
            ],
        )

        candidates = orchestrator.fetch(
            mode=IngestMode.DAILY_WATCH,
            feed_urls=["https://rss.arxiv.org/rss/cs.CV"],
            rolling_window_days=2,
        )

        self.assertEqual([candidate.title for candidate in candidates], ["rss:https://rss.arxiv.org/rss/cs.CV", "recent-new:https://rss.arxiv.org/rss/cs.CV"])

    def test_daily_watch_raises_when_all_feeds_fail(self):
        def failing_fetcher(feed_url, *, session=None):
            raise RuntimeError(f"boom:{feed_url}")

        orchestrator = IngestOrchestrator(rss_candidate_fetcher=failing_fetcher)

        with self.assertRaises(RuntimeError):
            orchestrator.fetch(
                mode=IngestMode.DAILY_WATCH,
                feed_urls=["https://rss.arxiv.org/rss/cs.CV"],
            )

    def test_daily_watch_with_rss_only_backend_skips_recent_fetch(self):
        recent_calls: list[tuple[int, str]] = []
        orchestrator = IngestOrchestrator(
            rss_candidate_fetcher=lambda feed_url, *, session=None: [_candidate("0001", "rss")],
            rolling_window_fetcher=lambda days, feed_url, *, session=None: recent_calls.append((days, feed_url)) or [],
        )

        candidates = orchestrator.fetch(
            mode=IngestMode.DAILY_WATCH,
            feed_urls=["https://rss.arxiv.org/rss/cs.CV"],
            rolling_window_days=5,
            backend_names=["rss"],
        )

        self.assertEqual([candidate.title for candidate in candidates], ["rss"])
        self.assertEqual(recent_calls, [])

    def test_daily_watch_with_explicit_arxiv_api_backend_uses_one_day_window_when_zero(self):
        orchestrator = IngestOrchestrator(
            rss_candidate_fetcher=lambda feed_url, *, session=None: [],
            rolling_window_fetcher=lambda days, feed_url, *, session=None: [_candidate(f"{days:04d}", f"recent:{days}")],
        )

        candidates = orchestrator.fetch(
            mode=IngestMode.DAILY_WATCH,
            feed_urls=["https://rss.arxiv.org/rss/cs.CV"],
            rolling_window_days=0,
            backend_names=["arxiv_api"],
        )

        self.assertEqual([candidate.title for candidate in candidates], ["recent:1"])

    def test_daily_watch_partial_rss_failure_keeps_surviving_feed_candidates(self):
        def rss_fetcher(feed_url, *, session=None):
            if feed_url.endswith("cs.RO"):
                raise RuntimeError("boom")
            return [_candidate("0001", f"rss:{feed_url}")]

        orchestrator = IngestOrchestrator(rss_candidate_fetcher=rss_fetcher)

        candidates = orchestrator.fetch(
            mode=IngestMode.DAILY_WATCH,
            feed_urls=["https://rss.arxiv.org/rss/cs.CV", "https://rss.arxiv.org/rss/cs.RO"],
        )

        self.assertEqual([candidate.title for candidate in candidates], ["rss:https://rss.arxiv.org/rss/cs.CV"])

    def test_daily_watch_recent_fetch_failure_falls_back_to_rss(self):
        orchestrator = IngestOrchestrator(
            rss_candidate_fetcher=lambda feed_url, *, session=None: [_candidate("0001", "rss")],
            rolling_window_fetcher=lambda days, feed_url, *, session=None: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        candidates = orchestrator.fetch(
            mode=IngestMode.DAILY_WATCH,
            feed_urls=["https://rss.arxiv.org/rss/cs.CV"],
            rolling_window_days=2,
        )

        self.assertEqual([candidate.title for candidate in candidates], ["rss"])

    def test_backfill_uses_arxiv_backend(self):
        class FakeArxivBackend:
            def fetch(self, **kwargs):
                self.kwargs = kwargs
                return [_candidate("0003", "backfill")]

        backend = FakeArxivBackend()
        orchestrator = IngestOrchestrator(arxiv_api_backend=backend)

        candidates = orchestrator.fetch(
            mode=IngestMode.BACKFILL,
            categories=["cs.CV"],
            start_dt=date(2026, 4, 1),
            end_dt=date(2026, 4, 2),
            max_results=25,
        )

        self.assertEqual([candidate.title for candidate in candidates], ["backfill"])
        self.assertEqual(backend.kwargs["categories"], ["cs.CV"])
        self.assertEqual(backend.kwargs["max_results"], 25)

    def test_backfill_requires_arxiv_api_backend_when_disabled(self):
        orchestrator = IngestOrchestrator()

        with self.assertRaisesRegex(ValueError, "requires ingest backend 'arxiv_api'"):
            orchestrator.fetch(
                mode=IngestMode.BACKFILL,
                categories=["cs.CV"],
                start_dt=date(2026, 4, 1),
                end_dt=date(2026, 4, 2),
                backend_names=["rss"],
            )

    def test_catch_up_requires_arxiv_api_backend_when_disabled(self):
        orchestrator = IngestOrchestrator()

        with self.assertRaisesRegex(ValueError, "requires ingest backend 'arxiv_api'"):
            orchestrator.fetch(
                mode=IngestMode.CATCH_UP,
                categories=["cs.CV"],
                backend_names=["rss"],
            )

    def test_catch_up_uses_sync_state_per_category_and_updates_progress(self):
        class FakeArxivBackend:
            def __init__(self):
                self.calls: list[dict] = []

            def fetch(self, **kwargs):
                self.calls.append(kwargs)
                category = kwargs["categories"][0]
                return [_candidate(f"{category}-001", f"catch-up:{category}")]

        updates: list[tuple[str, datetime, datetime, int]] = []
        backend = FakeArxivBackend()
        orchestrator = IngestOrchestrator(
            arxiv_api_backend=backend,
            sync_state_reader=lambda categories: {
                "cs.CV": datetime(2026, 1, 7, 12, 0, 0),
                "cs.LG": datetime(2026, 1, 9, 9, 30, 0),
            },
            sync_state_writer=lambda category, *, synced_through, updated_at, paper_count: updates.append(
                (category, synced_through, updated_at, paper_count)
            ),
            clock=lambda: datetime(2026, 1, 15, 8, 0, 0),
        )

        candidates = orchestrator.fetch(
            mode=IngestMode.CATCH_UP,
            categories=["cs.CV", "cs.LG"],
            end_dt=date(2026, 1, 15),
            max_results=25,
        )

        self.assertEqual([candidate.title for candidate in candidates], ["catch-up:cs.CV", "catch-up:cs.LG"])
        self.assertEqual(
            backend.calls,
            [
                {
                    "categories": ["cs.CV"],
                    "start_dt": date(2026, 1, 7),
                    "end_dt": date(2026, 1, 15),
                    "max_results": 25,
                    "session": None,
                },
                {
                    "categories": ["cs.LG"],
                    "start_dt": date(2026, 1, 9),
                    "end_dt": date(2026, 1, 15),
                    "max_results": 25,
                    "session": None,
                },
            ],
        )
        self.assertEqual(
            updates,
            [
                ("cs.CV", datetime(2026, 1, 15, 23, 59, 59, 999999), datetime(2026, 1, 15, 8, 0, 0), 1),
                ("cs.LG", datetime(2026, 1, 15, 23, 59, 59, 999999), datetime(2026, 1, 15, 8, 0, 0), 1),
            ],
        )

    def test_catch_up_requires_sync_state_cursor_for_each_category(self):
        orchestrator = IngestOrchestrator(
            sync_state_reader=lambda categories: {"cs.CV": datetime(2026, 1, 7, 12, 0, 0)},
        )

        with self.assertRaisesRegex(ValueError, "requires SyncState.last_synced_submitted_at"):
            orchestrator.fetch(
                mode=IngestMode.CATCH_UP,
                categories=["cs.CV", "cs.LG"],
            )
