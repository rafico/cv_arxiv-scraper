from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from app.services.arxiv_adapter import result_to_entry
from app.services.enrichment import parse_feed_entries, query_arxiv_api
from app.services.ingest import ArxivApiBackend, PaperCandidate, RssFeedBackend

RSS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test feed</title>
    <item>
      <title>Vision Paper</title>
      <link>https://arxiv.org/abs/2604.00001v1</link>
      <author>Alice Example and Bob Example</author>
      <description><![CDATA[<p>Useful abstract.</p>]]></description>
      <pubDate>Wed, 01 Apr 2026 12:34:56 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

ARXIV_API_XML_PAGE_ONE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>https://arxiv.org/abs/2604.00002v1</id>
    <published>2026-04-01T08:00:00Z</published>
    <title>API Paper</title>
    <summary>Abstract from API</summary>
    <author><name>Carol Example</name></author>
    <category term="cs.CV" />
    <arxiv:comment>Project page: https://example.com/project</arxiv:comment>
    <arxiv:doi>10.48550/arXiv.2604.00002</arxiv:doi>
  </entry>
</feed>
"""

ARXIV_API_XML_RESUME_FIRST_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2604.00003v1</id>
    <published>2026-04-01T08:00:00Z</published>
    <title>Paper 3</title>
    <summary>Abstract 3</summary>
    <author><name>Author A</name></author>
    <category term="cs.CV" />
  </entry>
  <entry>
    <id>https://arxiv.org/abs/2604.00004v1</id>
    <published>2026-04-01T08:00:00Z</published>
    <title>Paper 4</title>
    <summary>Abstract 4</summary>
    <author><name>Author A</name></author>
    <category term="cs.CV" />
  </entry>
</feed>
"""

ARXIV_API_XML_RESUME_SECOND_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2604.00005v1</id>
    <published>2026-04-01T08:00:00Z</published>
    <title>Paper 5</title>
    <summary>Abstract 5</summary>
    <author><name>Author A</name></author>
    <category term="cs.CV" />
  </entry>
</feed>
"""


class PaperCandidateTests(TestCase):
    def test_round_trips_through_legacy_entry_dict(self):
        candidate = PaperCandidate(
            arxiv_id="2604.00001",
            link="https://arxiv.org/abs/2604.00001",
            title="Vision Paper",
            author="Alice Example",
            authors_list=["Alice Example"],
            abstract="Useful abstract.",
            publication_date="2026-04-01",
            categories=["cs.CV"],
        )

        restored = PaperCandidate.from_entry_dict(candidate.to_entry_dict())

        self.assertEqual(restored, candidate)


class RssFeedBackendTests(TestCase):
    @patch("app.services.ingest.rss_backend.request_with_backoff")
    def test_fetch_parses_feed_entries_into_candidates(self, mock_request):
        mock_request.return_value = Mock(content=RSS_XML)

        backend = RssFeedBackend(["https://rss.arxiv.org/rss/cs.CV"])
        candidates = backend.fetch()

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].arxiv_id, "2604.00001")
        self.assertEqual(candidates[0].title, "Vision Paper")
        self.assertEqual(candidates[0].authors_list, ["Alice Example and Bob Example"])
        self.assertEqual(candidates[0].publication_date, "2026-04-01")

    @patch("app.services.ingest.rss_backend.RssFeedBackend.fetch")
    def test_parse_feed_entries_preserves_legacy_dict_shape(self, mock_fetch):
        mock_fetch.return_value = [
            PaperCandidate(
                arxiv_id="2604.00001",
                link="https://arxiv.org/abs/2604.00001",
                title="Vision Paper",
                author="Alice Example",
                authors_list=["Alice Example"],
            )
        ]

        entries = parse_feed_entries("https://rss.arxiv.org/rss/cs.CV")

        self.assertEqual(entries[0]["title"], "Vision Paper")
        self.assertEqual(entries[0]["authors_list"], ["Alice Example"])
        self.assertIn("resource_links", entries[0])


class ArxivApiBackendTests(TestCase):
    @patch("app.services.ingest.arxiv_api_backend.request_with_backoff")
    def test_fetch_builds_submitted_date_query_and_returns_candidates(self, mock_request):
        mock_request.return_value = Mock(text=ARXIV_API_XML_PAGE_ONE)
        backend = ArxivApiBackend()
        candidates = backend.fetch(
            categories=["cs.CV", "cs.LG"],
            start_dt=date(2026, 4, 1),
            end_dt=date(2026, 4, 2),
            max_results=25,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].arxiv_id, "2604.00002")
        self.assertEqual(candidates[0].doi, "10.48550/arXiv.2604.00002")
        self.assertEqual(
            mock_request.call_args.kwargs["params"]["search_query"],
            "(cat:cs.CV OR cat:cs.LG) AND submittedDate:[202604010000 TO 202604022359]",
        )
        self.assertEqual(mock_request.call_args.kwargs["params"]["max_results"], 25)
        self.assertEqual(mock_request.call_args.kwargs["params"]["start"], 0)

    @patch("app.services.ingest.arxiv_api_backend.ArxivApiBackend.fetch")
    def test_query_arxiv_api_preserves_legacy_dict_shape(self, mock_fetch):
        mock_fetch.return_value = [
            PaperCandidate(
                arxiv_id="2604.00002",
                link="https://arxiv.org/abs/2604.00002",
                title="API Paper",
                author="Carol Example",
                authors_list=["Carol Example"],
                comment="Demo comment",
                doi="10.48550/arXiv.2604.00002",
            )
        ]

        entries = query_arxiv_api(["cs.CV"], date(2026, 4, 1), date(2026, 4, 2), max_results=25)

        self.assertEqual(entries[0]["title"], "API Paper")
        self.assertEqual(entries[0]["doi"], "10.48550/arXiv.2604.00002")
        self.assertIn("categories", entries[0])

    @patch("app.services.ingest.arxiv_api_backend.request_with_backoff")
    def test_fetch_resumes_from_offset_and_skips_processed_cursor(self, mock_request):
        mock_request.side_effect = [
            Mock(text=ARXIV_API_XML_RESUME_FIRST_PAGE),
            Mock(text=ARXIV_API_XML_RESUME_SECOND_PAGE),
        ]
        progress: list[tuple[int, str | None]] = []

        backend = ArxivApiBackend(page_size=2)
        candidates = backend.fetch(
            categories=["cs.CV"],
            start_dt=date(2026, 4, 1),
            end_dt=date(2026, 4, 2),
            max_results=25,
            offset=2,
            resume_after_arxiv_id="2604.00004",
            progress_callback=lambda page_number, candidate: progress.append((page_number, candidate.arxiv_id)),
        )

        self.assertEqual([candidate.arxiv_id for candidate in candidates], ["2604.00005"])
        self.assertEqual(progress, [(3, "2604.00005")])
        self.assertEqual(mock_request.call_args_list[0].kwargs["params"]["start"], 2)
        self.assertEqual(mock_request.call_args_list[1].kwargs["params"]["start"], 4)


class ArxivAdapterTests(TestCase):
    def test_result_to_entry_uses_candidate_shape(self):
        result = SimpleNamespace(
            entry_id="https://arxiv.org/abs/2604.00003v2",
            title="Adapter Paper",
            authors=[SimpleNamespace(name="Dana Example")],
            published=datetime(2026, 4, 3, 10, 0, 0),
            summary="Adapter abstract",
            categories=["cs.CV"],
            comment="Code: https://example.com/code",
            doi="10.48550/arXiv.2604.00003",
        )

        entry = result_to_entry(result)

        self.assertEqual(entry["arxiv_id"], "2604.00003")
        self.assertEqual(entry["author"], "Dana Example")
        self.assertEqual(entry["categories"], ["cs.CV"])
