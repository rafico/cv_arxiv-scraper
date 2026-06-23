from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from app.services.http_client import create_session, request_with_backoff, resolve_user_agent
from app.services.rate_limiter import TokenBucketRateLimiter, resolve_rate_limit_settings


class RateLimiterTests(unittest.TestCase):
    def test_token_bucket_waits_once_bucket_is_empty(self):
        now = [0.0]
        sleeps: list[float] = []

        def fake_time() -> float:
            return now[0]

        def fake_sleep(delay: float) -> None:
            sleeps.append(delay)
            now[0] += delay

        limiter = TokenBucketRateLimiter(
            requests_per_second=2.0,
            burst=2,
            time_fn=fake_time,
            sleep_fn=fake_sleep,
        )

        self.assertEqual(limiter.acquire(), 0.0)
        self.assertEqual(limiter.acquire(), 0.0)
        waited = limiter.acquire()

        self.assertAlmostEqual(waited, 0.5, places=6)
        self.assertEqual(sleeps, [0.5])

    def test_bulk_profile_caps_configured_rate(self):
        scraper_config = {
            "ingest": {
                "rate_limit": {
                    "requests_per_second": 5.0,
                    "burst": 4,
                }
            }
        }

        interactive = resolve_rate_limit_settings(scraper_config, profile="interactive")
        bulk = resolve_rate_limit_settings(scraper_config, profile="bulk")

        self.assertEqual(interactive.requests_per_second, 5.0)
        self.assertEqual(interactive.burst, 4)
        self.assertLessEqual(bulk.requests_per_second, 1.0 / 3.0)
        self.assertEqual(bulk.burst, 1)


class HttpClientTests(unittest.TestCase):
    def test_create_session_sets_user_agent_from_config(self):
        session = create_session(
            scraper_config={
                "ingest": {
                    "user_agent": "cv-arxiv-scraper/1.0 (test@example.com)",
                }
            }
        )
        self.addCleanup(session.close)

        self.assertEqual(session.headers["User-Agent"], "cv-arxiv-scraper/1.0 (test@example.com)")

    @patch("app.services.http_client.requests.request")
    def test_request_with_backoff_injects_user_agent_without_session(self, mock_request):
        response = Mock(spec=requests.Response)
        response.raise_for_status.return_value = None
        mock_request.return_value = response

        request_with_backoff(
            "GET",
            "https://example.invalid/data",
            scraper_config={"ingest": {"user_agent": "cv-arxiv-scraper/1.0 (test@example.com)"}},
        )

        self.assertEqual(
            mock_request.call_args.kwargs["headers"]["User-Agent"],
            "cv-arxiv-scraper/1.0 (test@example.com)",
        )

    def test_resolve_user_agent_falls_back_to_default(self):
        self.assertTrue(resolve_user_agent({}).startswith("cv-arxiv-scraper/"))

    def test_request_with_backoff_reconfigures_session_for_bulk_profile(self):
        session = create_session(
            scraper_config={
                "ingest": {
                    "rate_limit": {
                        "requests_per_second": 5.0,
                        "burst": 4,
                    }
                }
            },
            rate_limit_profile="interactive",
        )
        self.addCleanup(session.close)

        response = Mock(spec=requests.Response)
        response.raise_for_status.return_value = None
        session.request = Mock(return_value=response)

        request_with_backoff(
            "GET",
            "https://example.invalid/data",
            session=session,
            scraper_config={
                "ingest": {
                    "rate_limit": {
                        "requests_per_second": 5.0,
                        "burst": 4,
                    }
                }
            },
            rate_limit_profile="bulk",
        )

        self.assertEqual(session._cv_arxiv_rate_limit_settings.profile, "bulk")


class RetryPolicyTests(unittest.TestCase):
    # A generous rate limit so the token bucket never sleeps and the tests stay fast;
    # only the retry/backoff behaviour is under test here.
    _FAST_CONFIG = {"ingest": {"rate_limit": {"requests_per_second": 1000.0, "burst": 1000}}}

    @staticmethod
    def _http_error(status: int) -> requests.HTTPError:
        response = Mock(spec=requests.Response)
        response.status_code = status
        error = requests.HTTPError(f"{status} error", response=response)
        response.raise_for_status.side_effect = error
        return response, error

    @patch("app.services.http_client.time.sleep")
    @patch("app.services.http_client.requests.request")
    def test_does_not_retry_non_retryable_client_error(self, mock_request, mock_sleep):
        response, _ = self._http_error(404)
        mock_request.return_value = response

        with self.assertRaises(requests.HTTPError):
            request_with_backoff("GET", "https://example.invalid/pdf", attempts=3, scraper_config=self._FAST_CONFIG)

        # 404 is permanent: one attempt, no retry, no backoff sleep.
        self.assertEqual(mock_request.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("app.services.http_client.time.sleep")
    @patch("app.services.http_client.requests.request")
    def test_retries_on_server_error(self, mock_request, mock_sleep):
        response, _ = self._http_error(503)
        mock_request.return_value = response

        with self.assertRaises(requests.HTTPError):
            request_with_backoff("GET", "https://example.invalid/data", attempts=3, scraper_config=self._FAST_CONFIG)

        self.assertEqual(mock_request.call_count, 3)

    @patch("app.services.http_client.time.sleep")
    @patch("app.services.http_client.requests.request")
    def test_retries_on_rate_limit(self, mock_request, mock_sleep):
        response, _ = self._http_error(429)
        mock_request.return_value = response

        with self.assertRaises(requests.HTTPError):
            request_with_backoff("GET", "https://example.invalid/data", attempts=2, scraper_config=self._FAST_CONFIG)

        self.assertEqual(mock_request.call_count, 2)

    @patch("app.services.http_client.time.sleep")
    @patch("app.services.http_client.requests.request")
    def test_retries_on_network_timeout(self, mock_request, mock_sleep):
        # Network-level errors carry no response and are transient → retried.
        mock_request.side_effect = requests.Timeout("read timed out")

        with self.assertRaises(requests.Timeout):
            request_with_backoff("GET", "https://example.invalid/data", attempts=2, scraper_config=self._FAST_CONFIG)

        self.assertEqual(mock_request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
