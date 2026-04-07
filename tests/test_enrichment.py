from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import Mock, patch

from app.services.enrichment import _fetch_api_metadata, fetch_recent_papers


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
        self.assertEqual(mock_request.call_args.kwargs["rate_limit_profile"], "bulk")

    @patch("app.services.enrichment.time.sleep")
    @patch("app.services.enrichment.request_with_backoff")
    def test_fetch_api_metadata_splits_failed_batch(self, mock_request, _mock_sleep):
        def _xml_for_id(arxiv_id: str) -> str:
            return f"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}v1</id>
    <author><name>Author</name><arxiv:affiliation>Test Lab</arxiv:affiliation></author>
    <category term="cs.CV" />
    <arxiv:comment>Has code</arxiv:comment>
    <arxiv:doi>10.1000/test</arxiv:doi>
  </entry>
</feed>"""

        def _side_effect(_method, _url, **kwargs):
            ids = kwargs["params"]["id_list"].split(",")
            if len(ids) > 1:
                raise RuntimeError("429")
            return Mock(text=_xml_for_id(ids[0]))

        mock_request.side_effect = _side_effect

        metadata = _fetch_api_metadata(["2604.00001", "2604.00002"])

        self.assertEqual(set(metadata), {"2604.00001", "2604.00002"})
        self.assertEqual(
            [len(call.kwargs["params"]["id_list"].split(",")) for call in mock_request.call_args_list],
            [2, 1, 1],
        )
        self.assertTrue(all(call.kwargs["rate_limit_profile"] == "bulk" for call in mock_request.call_args_list))


if __name__ == "__main__":
    unittest.main()
