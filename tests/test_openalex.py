"""Tests for OpenAlex enrichment service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.openalex import _parse_openalex_work, fetch_openalex_batch


class TestParseOpenalexWork:
    def test_full_work(self):
        work = {
            "id": "https://openalex.org/W12345",
            "doi": "https://doi.org/10.48550/arXiv.2301.00001",
            "open_access": {"oa_status": "gold", "is_oa": True},
            "cited_by_count": 42,
            "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"],
            "topics": [
                {"display_name": "Computer Vision", "score": 0.95},
                {"display_name": "Deep Learning", "score": 0.80},
            ],
        }
        result = _parse_openalex_work(work)
        assert result["openalex_id"] == "W12345"
        assert result["oa_status"] == "gold"
        assert result["openalex_cited_by_count"] == 42
        assert result["referenced_works_count"] == 2
        assert len(result["openalex_topics"]) == 2
        assert result["openalex_topics"][0]["name"] == "Computer Vision"

    def test_minimal_work(self):
        work = {"id": "", "topics": [], "open_access": {}, "cited_by_count": 0}
        result = _parse_openalex_work(work)
        assert result["openalex_topics"] == []
        assert result["oa_status"] is None
        assert result["openalex_cited_by_count"] == 0
        assert result["referenced_works_count"] == 0

    def test_missing_open_access(self):
        work = {"id": "https://openalex.org/W1", "topics": []}
        result = _parse_openalex_work(work)
        assert result["oa_status"] is None


class TestFetchOpenalexBatch:
    def test_empty_ids(self):
        assert fetch_openalex_batch([]) == {}

    @patch("app.services.openalex.request_with_backoff")
    def test_successful_batch(self, mock_request):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "https://openalex.org/W12345",
                    "doi": "https://doi.org/10.48550/arxiv.2301.00001",
                    "open_access": {"oa_status": "green"},
                    "cited_by_count": 10,
                    "referenced_works": [],
                    "topics": [{"display_name": "NLP", "score": 0.9}],
                }
            ]
        }
        mock_request.return_value = mock_response

        result = fetch_openalex_batch(["2301.00001"])
        assert "2301.00001" in result
        assert result["2301.00001"]["oa_status"] == "green"
        assert result["2301.00001"]["openalex_cited_by_count"] == 10

    @patch("app.services.openalex.request_with_backoff")
    def test_failed_request(self, mock_request):
        mock_request.return_value = None
        result = fetch_openalex_batch(["2301.00001"])
        assert result == {}

    @patch("app.services.openalex.request_with_backoff")
    def test_exception_handling(self, mock_request):
        mock_request.side_effect = Exception("Network error")
        result = fetch_openalex_batch(["2301.00001"])
        assert result == {}

    @patch("app.services.openalex.request_with_backoff")
    def test_email_parameter(self, mock_request):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_request.return_value = mock_response

        fetch_openalex_batch(["2301.00001"], email="test@example.com")
        call_kwargs = mock_request.call_args
        assert "mailto" in call_kwargs.kwargs.get("params", call_kwargs[1].get("params", {}))
