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

    def test_explicit_null_fields_do_not_raise(self):
        # OpenAlex returns explicit null for absent fields, so .get(key, default)
        # still yields None. The parser must coerce instead of crashing.
        work = {
            "id": None,
            "doi": None,
            "topics": None,
            "open_access": None,
            "referenced_works": None,
            "cited_by_count": None,
        }
        result = _parse_openalex_work(work)
        assert result["openalex_id"] == ""
        assert result["openalex_topics"] == []
        assert result["oa_status"] is None
        assert result["referenced_works_count"] == 0


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
    def test_prefix_ids_are_not_misattributed(self, mock_request):
        # The batch holds a short id that is a string-prefix of a longer one. The
        # only returned work belongs to the longer id and must NOT be attributed
        # to the shorter id via a substring match.
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "https://openalex.org/W999",
                    "doi": "https://doi.org/10.48550/arxiv.2301.00012",
                    "open_access": {"oa_status": "green"},
                    "cited_by_count": 99,
                    "referenced_works": [],
                    "topics": [],
                }
            ]
        }
        mock_request.return_value = mock_response

        result = fetch_openalex_batch(["2301.0001", "2301.00012"])
        assert "2301.0001" not in result
        assert result["2301.00012"]["openalex_cited_by_count"] == 99

    @patch("app.services.openalex.request_with_backoff")
    def test_versioned_doi_still_matches(self, mock_request):
        # A DOI carrying a trailing version (v2) must still map to the bare id.
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "doi": "https://doi.org/10.48550/arxiv.2301.00001v2",
                    "open_access": {"oa_status": "green"},
                    "cited_by_count": 7,
                    "referenced_works": [],
                    "topics": [],
                }
            ]
        }
        mock_request.return_value = mock_response

        result = fetch_openalex_batch(["2301.00001"])
        assert result["2301.00001"]["openalex_cited_by_count"] == 7

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
    def test_one_malformed_work_does_not_abort_batch(self, mock_request):
        # A non-dict (or otherwise broken) work in results must be skipped, not
        # abandon the rest of the 50-paper batch.
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                "not-a-dict",  # raises AttributeError on work.get(...)
                {
                    "id": "https://openalex.org/W999",
                    "doi": "https://doi.org/10.48550/arxiv.2301.00002",
                    "open_access": {"oa_status": "gold"},
                    "cited_by_count": 3,
                    "referenced_works": [],
                    "topics": [],
                },
            ]
        }
        mock_request.return_value = mock_response

        result = fetch_openalex_batch(["2301.00001", "2301.00002"])
        assert "2301.00002" in result
        assert result["2301.00002"]["oa_status"] == "gold"

    @patch("app.services.openalex.request_with_backoff")
    def test_email_parameter(self, mock_request):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_request.return_value = mock_response

        fetch_openalex_batch(["2301.00001"], email="test@example.com")
        call_kwargs = mock_request.call_args
        assert "mailto" in call_kwargs.kwargs.get("params", call_kwargs[1].get("params", {}))
