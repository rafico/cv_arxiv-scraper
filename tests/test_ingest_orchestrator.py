from __future__ import annotations

from datetime import date
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
