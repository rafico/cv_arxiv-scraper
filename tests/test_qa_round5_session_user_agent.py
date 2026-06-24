"""QA round 5 regression test — R5-1 (S2): the live scrape session's configured
User-Agent / rate-limit settings must survive a bare ``request_with_backoff``
call that reuses the session without re-passing ``scraper_config``/``user_agent``.

The RSS ingest backend (rss_backend.py) calls ``request_with_backoff("GET",
feed_url, timeout=30, session=session)`` — no config. Pre-fix, that made the
"requested" UA resolve to the library default, which mismatched the session's
configured UA, tripping the reconfigure branch and resetting the session's
User-Agent and rate-limit settings to interactive defaults for the rest of the
scrape — silently dropping a configured ``ingest.user_agent`` on the default
daily/scheduled scrape path.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from app.services.http_client import create_session, request_with_backoff
from app.services.ingest.rss_backend import RssFeedBackend

_UA = "MyApp/9.9 (mailto:me@example.com)"


class BareCallPreservesSessionConfigTests(unittest.TestCase):
    def test_bare_request_keeps_configured_user_agent_and_rate_limit(self):
        session = create_session(
            scraper_config={
                "ingest": {
                    "user_agent": _UA,
                    "rate_limit": {"requests_per_second": 5.0, "burst": 4},
                }
            },
            rate_limit_profile="bulk",
        )
        self.addCleanup(session.close)

        response = Mock(spec=requests.Response)
        response.raise_for_status.return_value = None
        session.request = Mock(return_value=response)

        # Reuse the session the way the RSS backend does: only session=, no config.
        request_with_backoff("GET", "https://example.invalid/feed", session=session)

        self.assertEqual(session.headers["User-Agent"], _UA)
        self.assertEqual(session._cv_arxiv_user_agent, _UA)
        # The bulk rate-limit profile must not have been reset to interactive defaults.
        self.assertEqual(session._cv_arxiv_rate_limit_settings.profile, "bulk")

    def test_bulk_profile_call_retunes_rate_limit_but_keeps_user_agent(self):
        # enrichment._request_arxiv_api passes rate_limit_profile="bulk" with no config
        # or user_agent: it must re-tune the throttle yet keep the configured UA.
        session = create_session(
            scraper_config={"ingest": {"user_agent": _UA}},
            rate_limit_profile="interactive",
        )
        self.addCleanup(session.close)

        response = Mock(spec=requests.Response)
        response.raise_for_status.return_value = None
        session.request = Mock(return_value=response)

        request_with_backoff("GET", "https://example.invalid/api", session=session, rate_limit_profile="bulk")

        self.assertEqual(session._cv_arxiv_rate_limit_settings.profile, "bulk")
        self.assertEqual(session.headers["User-Agent"], _UA)

    def test_user_agent_only_call_keeps_configured_rate_limit(self):
        # A call that overrides only the user_agent must not reset the rate limit.
        session = create_session(
            scraper_config={"ingest": {"rate_limit": {"requests_per_second": 5.0, "burst": 4}}},
            rate_limit_profile="bulk",
        )
        self.addCleanup(session.close)
        before = session._cv_arxiv_rate_limit_settings

        response = Mock(spec=requests.Response)
        response.raise_for_status.return_value = None
        session.request = Mock(return_value=response)

        request_with_backoff("GET", "https://example.invalid/api", session=session, user_agent="Other/1.0")

        self.assertEqual(session._cv_arxiv_rate_limit_settings, before)
        self.assertEqual(session.headers["User-Agent"], "Other/1.0")


class RssBackendDoesNotResetSessionTests(unittest.TestCase):
    @patch("app.services.ingest.rss_backend.feedparser.parse")
    def test_rss_fetch_preserves_session_user_agent(self, mock_parse):
        mock_parse.return_value = Mock(entries=[])
        session = create_session(scraper_config={"ingest": {"user_agent": _UA}})
        self.addCleanup(session.close)

        response = Mock(spec=requests.Response)
        response.raise_for_status.return_value = None
        response.content = b""
        session.request = Mock(return_value=response)

        RssFeedBackend(["https://export.arxiv.org/rss/cs.CV"]).fetch(session=session)

        self.assertEqual(session.request.call_args.args[0], "GET")
        self.assertEqual(session.headers["User-Agent"], _UA)


if __name__ == "__main__":
    unittest.main()
