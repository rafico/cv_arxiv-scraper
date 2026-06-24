"""QA round 5 regression test — R5-con3 (S2): a mid-pagination arXiv failure (or
malformed page) in fetch_recent_papers must keep the entries already paginated this
run, not discard them by unwinding the whole call.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from app.services import enrichment


def _atom_page(arxiv_ids: list[str]) -> str:
    entries = "".join(
        f"<entry><id>http://arxiv.org/abs/{aid}</id><title>T {aid}</title>"
        f'<link href="http://arxiv.org/abs/{aid}"/><published>2026-01-01T00:00:00Z</published>'
        f"<summary>abstract</summary></entry>"
        for aid in arxiv_ids
    )
    return f'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:atom="http://www.w3.org/2005/Atom">{entries}</feed>'


class FetchRecentPaginationTests(unittest.TestCase):
    @patch.object(enrichment, "query_arxiv_api", return_value=[])
    @patch.object(enrichment, "_ARXIV_ROLLING_WINDOW_MAX_PAGES", 5)
    @patch.object(enrichment, "_ARXIV_API_BATCH_SIZE", 2)
    @patch("app.services.enrichment.time.sleep", lambda *_a, **_k: None)
    @patch.object(enrichment, "_request_arxiv_api")
    def test_midpagination_failure_keeps_collected_entries(self, mock_request, _mock_query):
        # First page: a full batch (forces a second page). Second page: network error.
        full_page = Mock(text=_atom_page(["2601.00001", "2601.00002"]))
        mock_request.side_effect = [full_page, RuntimeError("arXiv 503 after retries")]

        entries = enrichment.fetch_recent_papers(2, "https://export.arxiv.org/rss/cs.CV")

        # The two entries from the first page survive the second page's failure.
        self.assertEqual(len(entries), 2)


if __name__ == "__main__":
    unittest.main()
