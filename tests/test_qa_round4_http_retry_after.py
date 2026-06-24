from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import Mock, patch

import requests

from app.services.http_client import request_with_backoff

# A generous rate limit so the token bucket never sleeps and the tests stay fast;
# only the Retry-After backoff behaviour is under test here.
_FAST_CONFIG = {"ingest": {"rate_limit": {"requests_per_second": 1000.0, "burst": 1000}}}


def _error_response(status: int, headers: dict[str, str] | None = None) -> requests.Response:
    response = Mock(spec=requests.Response)
    response.status_code = status
    response.headers = headers or {}
    error = requests.HTTPError(f"{status} error", response=response)
    response.raise_for_status.side_effect = error
    return response


def _ok_response() -> requests.Response:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.headers = {}
    response.raise_for_status.return_value = None
    return response


class RetryAfterBackoffTests(unittest.TestCase):
    """G19: request_with_backoff must honour the Retry-After header on 429/503."""

    @patch("app.services.http_client.time.sleep")
    @patch("app.services.http_client.requests.request")
    def test_g19_honours_integer_retry_after_seconds(self, mock_request, mock_sleep):
        # 429 with Retry-After: 30, then a success.
        mock_request.side_effect = [
            _error_response(429, {"Retry-After": "30"}),
            _ok_response(),
        ]

        response = request_with_backoff(
            "GET",
            "https://example.invalid/data",
            attempts=3,
            scraper_config=_FAST_CONFIG,
        )

        self.assertEqual(response.status_code, 200)
        # The computed backoff for attempt 1 is base_delay * 2**0 == 1.25; the
        # server asked for 30, so we must sleep at least 30, not 1.25.
        self.assertEqual(mock_sleep.call_count, 1)
        slept = mock_sleep.call_args.args[0]
        self.assertGreaterEqual(slept, 30.0)
        self.assertLessEqual(slept, 120.0)

    @patch("app.services.http_client.time.sleep")
    @patch("app.services.http_client.requests.request")
    def test_g19_honours_http_date_retry_after(self, mock_request, mock_sleep):
        future = datetime.now(timezone.utc) + timedelta(seconds=45)
        http_date = format_datetime(future, usegmt=True)
        mock_request.side_effect = [
            _error_response(503, {"Retry-After": http_date}),
            _ok_response(),
        ]

        response = request_with_backoff(
            "GET",
            "https://example.invalid/data",
            attempts=3,
            scraper_config=_FAST_CONFIG,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_sleep.call_count, 1)
        slept = mock_sleep.call_args.args[0]
        # Allow a little slack for clock drift between formatting and parsing.
        self.assertGreaterEqual(slept, 40.0)
        self.assertLessEqual(slept, 120.0)

    @patch("app.services.http_client.time.sleep")
    @patch("app.services.http_client.requests.request")
    def test_g19_clamps_excessive_retry_after(self, mock_request, mock_sleep):
        mock_request.side_effect = [
            _error_response(429, {"Retry-After": "9999"}),
            _ok_response(),
        ]

        request_with_backoff(
            "GET",
            "https://example.invalid/data",
            attempts=3,
            scraper_config=_FAST_CONFIG,
        )

        slept = mock_sleep.call_args.args[0]
        self.assertLessEqual(slept, 120.0)

    @patch("app.services.http_client.time.sleep")
    @patch("app.services.http_client.requests.request")
    def test_g19_falls_back_to_computed_backoff_for_garbage_header(self, mock_request, mock_sleep):
        mock_request.side_effect = [
            _error_response(429, {"Retry-After": "not-a-date"}),
            _ok_response(),
        ]

        request_with_backoff(
            "GET",
            "https://example.invalid/data",
            attempts=3,
            scraper_config=_FAST_CONFIG,
        )

        # Garbage header → computed backoff (1.25) is used.
        slept = mock_sleep.call_args.args[0]
        self.assertAlmostEqual(slept, 1.25, places=6)


if __name__ == "__main__":
    unittest.main()
