from __future__ import annotations

import unittest
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.services.arxiv_adapter import result_to_entry
from app.services.enrichment import parse_feed_entries, query_arxiv_api
from app.services.ingest import PaperCandidate
from app.services.ingest.arxiv_api_backend import ArxivApiBackend
from app.services.ingest.rss_backend import RssFeedBackend


class PaperCandidateTests(unittest.TestCase):
    def test_from_entry_dict_round_trips_expected_fields(self):
        entry = {
            "arxiv_id": "1234.5678",
            "link": "https://arxiv.org/abs/1234.5678",
            "title": "A Paper",
            "author": "Author A",
            "authors_list": ["Author A"],
            "abstract": "Abstract",
            "published": "2026-04-01T12:00:00",
            "publication_dt": date(2026, 4, 1),
            "publication_date": "2026-04-01",
            "categories": ["cs.CV"],
            "comment": "conference version",
            "doi": "10.1000/test",
            "api_affiliations": "Example University",
            "resource_links": [{"type": "code", "url": "https://example.com"}],
        }

        candidate = PaperCandidate.from_entry_dict(entry)

        self.assertEqual(candidate.to_entry_dict(), entry)


class RssBackendTests(unittest.TestCase):
    @patch("app.services.ingest.rss_backend.request_with_backoff")
    def test_fetch_normalizes_feed_entries(self, mock_request):
        mock_request.return_value = Mock(
            content=b"""
            <rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
              <channel>
                <item>
                  <title>  Vision   Paper </title>
                  <link>https://arxiv.org/abs/1234.5678v2</link>
                  <description><![CDATA[<p>Test <b>abstract</b>.</p>]]></description>
                  <author>Author One and Author Two</author>
                  <pubDate>Tue, 01 Apr 2026 12:00:00 GMT</pubDate>
                </item>
              </channel>
            </rss>
            """
        )

        candidates = RssFeedBackend(["https://rss.arxiv.org/rss/cs.CV"]).fetch()

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].arxiv_id, "1234.5678")
        self.assertEqual(candidates[0].title, "Vision Paper")
        self.assertEqual(candidates[0].authors_list, ["Author One and Author Two"])
        self.assertEqual(candidates[0].abstract, "Test abstract .")
        self.assertEqual(candidates[0].publication_date, "2026-04-01")


class ArxivApiBackendTests(unittest.TestCase):
    def test_result_to_candidate_preserves_entry_shape(self):
        result = SimpleNamespace(
            entry_id="https://arxiv.org/abs/2501.00001v3",
            title="  Better Vision  ",
            summary="<p>Useful summary</p>",
            authors=[SimpleNamespace(name="Author A"), SimpleNamespace(name="Author B")],
            published=datetime(2026, 4, 1, 10, 30, 0),
            categories=["cs.CV", "cs.AI"],
            comment="accepted at conf",
            doi="10.1000/example",
        )

        candidate = ArxivApiBackend.result_to_candidate(result)

        self.assertEqual(candidate.arxiv_id, "2501.00001")
        self.assertEqual(candidate.author, "Author A, Author B")
        self.assertEqual(candidate.categories, ["cs.CV", "cs.AI"])
        self.assertEqual(candidate.comment, "accepted at conf")
        self.assertEqual(candidate.doi, "10.1000/example")

    def test_arxiv_adapter_keeps_legacy_dict_shape(self):
        result = SimpleNamespace(
            entry_id="https://arxiv.org/abs/2501.00002",
            title="Example",
            summary="Summary",
            authors=[SimpleNamespace(name="Author A")],
            published=datetime(2026, 4, 1, 10, 30, 0),
            categories=["cs.CV"],
            comment="",
            doi="",
        )

        entry = result_to_entry(result)

        self.assertEqual(entry["arxiv_id"], "2501.00002")
        self.assertEqual(entry["authors_list"], ["Author A"])
        self.assertEqual(entry["categories"], ["cs.CV"])


class EnrichmentCompatibilityTests(unittest.TestCase):
    @patch("app.services.enrichment.RssFeedBackend.fetch")
    def test_parse_feed_entries_returns_dicts(self, mock_fetch):
        mock_fetch.return_value = [
            PaperCandidate(
                arxiv_id="1234.5678",
                link="https://arxiv.org/abs/1234.5678",
                title="Paper",
            )
        ]

        entries = parse_feed_entries("https://rss.arxiv.org/rss/cs.CV")

        self.assertEqual(
            entries,
            [
                {
                    "arxiv_id": "1234.5678",
                    "link": "https://arxiv.org/abs/1234.5678",
                    "title": "Paper",
                    "author": "",
                    "authors_list": [],
                    "abstract": "",
                    "published": None,
                    "publication_dt": None,
                    "publication_date": "Date Unknown",
                    "categories": [],
                    "comment": "",
                    "doi": "",
                    "api_affiliations": "",
                    "resource_links": [],
                }
            ],
        )

    @patch("app.services.enrichment.ArxivApiBackend.fetch")
    def test_query_arxiv_api_returns_dicts(self, mock_fetch):
        mock_fetch.return_value = [
            PaperCandidate(
                arxiv_id="1234.5678",
                link="https://arxiv.org/abs/1234.5678",
                title="Paper",
                categories=["cs.CV"],
            )
        ]

        entries = query_arxiv_api(["cs.CV"], date(2026, 4, 1), date(2026, 4, 2), max_results=25)

        self.assertEqual(entries[0]["categories"], ["cs.CV"])
        mock_fetch.assert_called_once_with(
            categories=["cs.CV"],
            start_dt=date(2026, 4, 1),
            end_dt=date(2026, 4, 2),
            max_results=25,
        )


if __name__ == "__main__":
    unittest.main()
