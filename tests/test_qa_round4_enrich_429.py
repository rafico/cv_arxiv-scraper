from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from app.services.enrichment import _fetch_api_metadata_batch


def _make_http_error(status_code: int) -> requests.HTTPError:
    response = Mock()
    response.status_code = status_code
    return requests.HTTPError(response=response)


class FetchApiMetadataBatch429Tests(unittest.TestCase):
    """G20: a 429 rate-limit must NOT trigger recursive batch halving.

    Splitting on 429 multiplies the request count during a rate-limit storm
    (20 -> 10 -> 5 ...), making the throttling strictly worse. The batch is
    best-effort metadata, so a 429 should log + return without recursing.
    """

    @patch("app.services.enrichment.time.sleep")
    @patch("app.services.enrichment._request_arxiv_api")
    def test_429_does_not_recurse_split(self, mock_request, _mock_sleep):
        mock_request.side_effect = _make_http_error(429)

        metadata: dict[str, dict] = {}
        _fetch_api_metadata_batch([f"2604.0000{i}" for i in range(8)], metadata)

        # Exactly one attempt: no recursive halving on a rate-limit error.
        self.assertEqual(mock_request.call_count, 1)
        self.assertEqual(metadata, {})

    @patch("app.services.enrichment.time.sleep")
    @patch("app.services.enrichment._request_arxiv_api")
    def test_non_429_error_still_splits(self, mock_request, _mock_sleep):
        # A non-rate-limit failure (e.g. transient/over-large batch) keeps the
        # existing split-and-retry behavior.
        mock_request.side_effect = _make_http_error(500)

        metadata: dict[str, dict] = {}
        _fetch_api_metadata_batch([f"2604.0000{i}" for i in range(8)], metadata)

        self.assertGreater(mock_request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
