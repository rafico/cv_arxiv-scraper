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

        self.assertEqual(getattr(session, "_cv_arxiv_rate_limit_settings").profile, "bulk")


if __name__ == "__main__":
    unittest.main()
