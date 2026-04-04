from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

from app.models import EnrichmentCache, Paper, db
from app.services.citations import fetch_citations_batch
from app.services.openalex import fetch_openalex_batch
from app.services.text import now_utc
from tests.helpers import FlaskDBTestCase


def _paper(arxiv_id: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        authors="Author A",
        link=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_link=f"https://arxiv.org/pdf/{arxiv_id}",
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=1.0,
        publication_date="2026-01-01",
        scraped_date="2026-01-01",
    )


class EnrichmentCacheTests(FlaskDBTestCase):
    @patch("app.services.citations.request_with_backoff")
    def test_citations_use_cache_after_first_fetch(self, mock_request):
        paper = _paper("2601.10001")
        db.session.add(paper)
        db.session.commit()

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "citationCount": 12,
                "influentialCitationCount": 4,
                "paperId": "ss-123",
            }
        ]
        mock_request.return_value = mock_response

        first = fetch_citations_batch(["2601.10001"])
        second = fetch_citations_batch(["2601.10001"])

        self.assertEqual(first["2601.10001"]["citation_count"], 12)
        self.assertEqual(second["2601.10001"]["semantic_scholar_id"], "ss-123")
        self.assertEqual(mock_request.call_count, 1)

        cache_row = EnrichmentCache.query.filter_by(paper_id=paper.id, source="semantic_scholar").one()
        self.assertEqual(cache_row.data["citation_count"], 12)

    @patch("app.services.openalex.request_with_backoff")
    def test_openalex_uses_cache_after_first_fetch(self, mock_request):
        paper = _paper("2601.10002")
        db.session.add(paper)
        db.session.commit()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "https://openalex.org/W12345",
                    "doi": "https://doi.org/10.48550/arxiv.2601.10002",
                    "open_access": {"oa_status": "green"},
                    "cited_by_count": 10,
                    "referenced_works": [],
                    "topics": [{"display_name": "NLP", "score": 0.9}],
                }
            ]
        }
        mock_request.return_value = mock_response

        first = fetch_openalex_batch(["2601.10002"], email="test@example.com")
        second = fetch_openalex_batch(["2601.10002"], email="test@example.com")

        self.assertEqual(first["2601.10002"]["openalex_id"], "W12345")
        self.assertEqual(second["2601.10002"]["oa_status"], "green")
        self.assertEqual(mock_request.call_count, 1)

        cache_row = EnrichmentCache.query.filter_by(paper_id=paper.id, source="openalex").one()
        self.assertEqual(cache_row.data["openalex_id"], "W12345")

    @patch("app.services.citations.request_with_backoff")
    def test_stale_citation_cache_is_refreshed(self, mock_request):
        paper = _paper("2601.10003")
        db.session.add(paper)
        db.session.commit()

        stale_row = EnrichmentCache(
            paper_id=paper.id,
            source="semantic_scholar",
            data={"citation_count": 3, "influential_citation_count": 1, "semantic_scholar_id": "old"},
            fetched_at=now_utc() - timedelta(days=10),
            ttl_hours=24,
        )
        db.session.add(stale_row)
        db.session.commit()

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "citationCount": 25,
                "influentialCitationCount": 6,
                "paperId": "new",
            }
        ]
        mock_request.return_value = mock_response

        refreshed = fetch_citations_batch(["2601.10003"])

        self.assertEqual(refreshed["2601.10003"]["citation_count"], 25)
        self.assertEqual(mock_request.call_count, 1)
        updated_row = EnrichmentCache.query.filter_by(paper_id=paper.id, source="semantic_scholar").one()
        self.assertEqual(updated_row.data["semantic_scholar_id"], "new")
