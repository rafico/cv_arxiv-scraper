from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import Mock, patch

from app.services.enrichment import fetch_recent_papers


class FetchRecentPapersTests(unittest.TestCase):
    @patch("app.services.enrichment.utc_today", return_value=date(2026, 3, 20))
    @patch("app.services.enrichment.request_with_backoff")
    def test_uses_utc_today_for_query_window(self, mock_request, _mock_today):
        mock_request.return_value = Mock(text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>')

        fetch_recent_papers(2, "https://rss.arxiv.org/rss/cs.CV")

        params = mock_request.call_args.kwargs["params"]
        self.assertEqual(
            params["search_query"],
            "cat:cs.CV AND submittedDate:[202603170000 TO 202603202359]",
        )


if __name__ == "__main__":
    unittest.main()
