"""Ecosystem-survival hardening tests.

Covers data-source API key resolution (env > dotfile > none), OpenAlex api_key
param + one-time missing-key warning, Semantic Scholar x-api-key header + bulk
pacing, GitHub token dotfile fallback, Retry-After handling in
request_with_backoff, and the Settings "Data Sources" / "Feed Sources" UI.
All HTTP is mocked.
"""

from __future__ import annotations

import os
import stat
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import requests

from app.services.enrichment_providers import GitHubProvider, OpenAlexProvider, SemanticScholarProvider
from app.services.http_client import request_with_backoff
from app.services.secret_files import (
    data_source_key_path,
    has_data_source_key,
    resolve_data_source_key,
    write_secret_file,
)
from tests.helpers import FlaskDBTestCase

# Neutralise any keys present in the developer's real environment; empty values
# are treated as unset by the resolver.
_NO_KEY_ENV = {
    "OPENALEX_API_KEY": "",
    "SEMANTIC_SCHOLAR_API_KEY": "",
    "GITHUB_TOKEN": "",
}


def _openalex_response(results: list | None = None) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"results": results or []}
    return response


class SecretResolutionTests(FlaskDBTestCase):
    """Env var beats dotfile beats nothing, with dotfiles rooted next to .llm_api_key."""

    def test_dotfiles_live_next_to_llm_api_key(self):
        expected_root = Path(self.app.config["LLM_KEY_PATH"]).parent
        self.assertEqual(data_source_key_path("openalex"), expected_root / ".openalex_api_key")
        self.assertEqual(data_source_key_path("semantic_scholar"), expected_root / ".s2_api_key")
        self.assertEqual(data_source_key_path("github"), expected_root / ".github_token")

    def test_env_var_wins_over_dotfile(self):
        write_secret_file(data_source_key_path("openalex"), "file-key")
        with patch.dict(os.environ, {"OPENALEX_API_KEY": "env-key"}):
            self.assertEqual(resolve_data_source_key("openalex"), "env-key")

    def test_dotfile_used_when_env_unset(self):
        write_secret_file(data_source_key_path("semantic_scholar"), "file-key")
        with patch.dict(os.environ, _NO_KEY_ENV):
            self.assertEqual(resolve_data_source_key("semantic_scholar"), "file-key")

    def test_none_when_neither_configured(self):
        with patch.dict(os.environ, _NO_KEY_ENV):
            self.assertIsNone(resolve_data_source_key("github"))
            self.assertFalse(has_data_source_key("github"))

    def test_empty_dotfile_counts_as_unconfigured(self):
        write_secret_file(data_source_key_path("openalex"), "   \n")
        with patch.dict(os.environ, _NO_KEY_ENV):
            self.assertIsNone(resolve_data_source_key("openalex"))


class OpenAlexKeyTests(FlaskDBTestCase):
    def test_explicit_api_key_sent_as_query_param(self):
        request_fn = MagicMock(return_value=_openalex_response())
        provider = OpenAlexProvider(request_fn=request_fn, api_key="oa-key")

        with patch.dict(os.environ, _NO_KEY_ENV):
            provider.fetch_batch(["2301.00001"], email="me@example.com")

        params = request_fn.call_args.kwargs["params"]
        self.assertEqual(params["api_key"], "oa-key")
        self.assertEqual(params["mailto"], "me@example.com")  # mailto kept alongside the key

    def test_api_key_resolved_from_dotfile(self):
        write_secret_file(data_source_key_path("openalex"), "dotfile-key")
        request_fn = MagicMock(return_value=_openalex_response())
        provider = OpenAlexProvider(request_fn=request_fn)

        with patch.dict(os.environ, _NO_KEY_ENV):
            provider.fetch_batch(["2301.00001"])

        self.assertEqual(request_fn.call_args.kwargs["params"]["api_key"], "dotfile-key")

    def test_no_key_omits_param_and_still_attempts_request(self):
        request_fn = MagicMock(return_value=_openalex_response())
        provider = OpenAlexProvider(request_fn=request_fn)

        with patch.dict(os.environ, _NO_KEY_ENV):
            result = provider.fetch_batch(["2301.00001"])

        self.assertEqual(request_fn.call_count, 1)
        self.assertNotIn("api_key", request_fn.call_args.kwargs["params"])
        self.assertEqual(result, {})

    def test_keyless_auth_failure_soft_fails_with_one_time_warning(self):
        response = Mock(status_code=429)
        error = requests.HTTPError("429 rate limited", response=response)
        request_fn = MagicMock(side_effect=error)
        provider = OpenAlexProvider(request_fn=request_fn)

        with (
            patch.dict(os.environ, _NO_KEY_ENV),
            patch("app.services.enrichment_providers.openalex_provider._missing_key_warned", False),
            self.assertLogs("app.services.enrichment_providers.openalex_provider", level="WARNING") as logs,
        ):
            first = provider.fetch_batch(["2301.00001"])
            second = provider.fetch_batch(["2301.00002"])

        self.assertEqual(first, {})  # soft-fail: no exception escapes
        self.assertEqual(second, {})
        hint_lines = [line for line in logs.output if "OpenAlex now requires an API key" in line]
        self.assertEqual(len(hint_lines), 1)  # emitted once, not per batch

    def test_auth_failure_with_key_does_not_emit_missing_key_hint(self):
        response = Mock(status_code=403)
        error = requests.HTTPError("403 forbidden", response=response)
        request_fn = MagicMock(side_effect=error)
        provider = OpenAlexProvider(request_fn=request_fn, api_key="oa-key")

        with (
            patch("app.services.enrichment_providers.openalex_provider._missing_key_warned", False),
            self.assertLogs("app.services.enrichment_providers.openalex_provider", level="WARNING") as logs,
        ):
            provider.fetch_batch(["2301.00001"])

        self.assertFalse(any("OpenAlex now requires an API key" in line for line in logs.output))


class SemanticScholarKeyTests(FlaskDBTestCase):
    @staticmethod
    def _capture_request_fn(calls: list[dict]):
        def request_fn(method, url, **kwargs):
            calls.append(kwargs)
            response = MagicMock()
            response.json.return_value = [None]
            return response

        return request_fn

    def test_x_api_key_header_injected_when_key_configured(self):
        calls: list[dict] = []
        provider = SemanticScholarProvider(request_fn=self._capture_request_fn(calls), api_key="s2-key")

        with patch.dict(os.environ, _NO_KEY_ENV):
            provider.fetch_batch(["2301.00001"])

        self.assertEqual(calls[0]["headers"], {"x-api-key": "s2-key"})

    def test_key_resolved_from_dotfile(self):
        write_secret_file(data_source_key_path("semantic_scholar"), "dotfile-s2")
        calls: list[dict] = []
        provider = SemanticScholarProvider(request_fn=self._capture_request_fn(calls))

        with patch.dict(os.environ, _NO_KEY_ENV):
            provider.fetch_batch(["2301.00001"])

        self.assertEqual(calls[0]["headers"], {"x-api-key": "dotfile-s2"})

    def test_env_var_beats_dotfile(self):
        write_secret_file(data_source_key_path("semantic_scholar"), "dotfile-s2")
        calls: list[dict] = []
        provider = SemanticScholarProvider(request_fn=self._capture_request_fn(calls))

        with patch.dict(os.environ, {**_NO_KEY_ENV, "SEMANTIC_SCHOLAR_API_KEY": "env-s2"}):
            provider.fetch_batch(["2301.00001"])

        self.assertEqual(calls[0]["headers"], {"x-api-key": "env-s2"})

    def test_no_key_keeps_legacy_request_fn_signature(self):
        # Injected doubles with the narrow keyword-only signature (no headers /
        # rate_limit_profile params) must keep working when no key is configured.
        seen: dict = {}

        def legacy_request_fn(method, url, *, json, params, session, timeout):
            seen["ids"] = json["ids"]
            response = MagicMock()
            response.json.return_value = [None]
            return response

        provider = SemanticScholarProvider(request_fn=legacy_request_fn)
        with patch.dict(os.environ, _NO_KEY_ENV):
            provider.fetch_batch(["2301.00001"])

        self.assertEqual(seen["ids"], ["ARXIV:2301.00001"])

    def test_real_client_gets_bulk_rate_limit_profile(self):
        # ~1 req/s S2 keys: the real request path must pace via the shared "bulk"
        # profile (≤ 1 request / 3 s), while injected doubles never see the kwarg.
        with patch("app.services.http_client.request_with_backoff") as mock_request:
            mock_request.return_value.json.return_value = [None]
            provider = SemanticScholarProvider()
            with patch.dict(os.environ, _NO_KEY_ENV):
                provider.fetch_batch(["2301.00001"])

        self.assertEqual(mock_request.call_args.kwargs["rate_limit_profile"], "bulk")


class GitHubTokenTests(FlaskDBTestCase):
    @staticmethod
    def _repo_response() -> MagicMock:
        response = MagicMock()
        response.json.return_value = {"full_name": "lab/model", "stargazers_count": 1, "license": None}
        return response

    def test_token_falls_back_to_dotfile(self):
        write_secret_file(data_source_key_path("github"), "gh-dotfile-token")
        request_fn = MagicMock(return_value=self._repo_response())
        provider = GitHubProvider(request_fn=request_fn)

        with patch.dict(os.environ, _NO_KEY_ENV):
            provider.fetch_batch(["2301.00001"], repos_by_arxiv_id={"2301.00001": "lab/model"})

        headers = request_fn.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer gh-dotfile-token")

    def test_explicit_token_beats_dotfile(self):
        write_secret_file(data_source_key_path("github"), "gh-dotfile-token")
        request_fn = MagicMock(return_value=self._repo_response())
        provider = GitHubProvider(request_fn=request_fn, token="explicit-token")

        with patch.dict(os.environ, _NO_KEY_ENV):
            provider.fetch_batch(["2301.00001"], repos_by_arxiv_id={"2301.00001": "lab/model"})

        headers = request_fn.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer explicit-token")

    def test_no_token_sends_no_auth_header(self):
        request_fn = MagicMock(return_value=self._repo_response())
        provider = GitHubProvider(request_fn=request_fn)

        with patch.dict(os.environ, _NO_KEY_ENV):
            provider.fetch_batch(["2301.00001"], repos_by_arxiv_id={"2301.00001": "lab/model"})

        self.assertNotIn("Authorization", request_fn.call_args.kwargs["headers"])


class RetryAfterTests(unittest.TestCase):
    """request_with_backoff must honor (bounded) Retry-After on 429/503."""

    # Generous limiter so the token bucket never sleeps; only backoff is under test.
    _FAST_CONFIG = {"ingest": {"rate_limit": {"requests_per_second": 1000.0, "burst": 1000}}}

    @staticmethod
    def _error_response(status: int, headers: dict | None = None) -> Mock:
        response = Mock(spec=requests.Response)
        response.status_code = status
        response.headers = headers or {}
        response.raise_for_status.side_effect = requests.HTTPError(f"{status} error", response=response)
        return response

    @patch("app.services.http_client.time.sleep")
    @patch("app.services.http_client.requests.request")
    def test_retry_after_seconds_honored_on_429(self, mock_request, mock_sleep):
        mock_request.return_value = self._error_response(429, {"Retry-After": "7"})

        with self.assertRaises(requests.HTTPError):
            request_with_backoff("GET", "https://example.invalid/x", attempts=2, scraper_config=self._FAST_CONFIG)

        self.assertEqual(mock_request.call_count, 2)
        mock_sleep.assert_called_once_with(7.0)  # max(exponential 1.25, Retry-After 7)

    @patch("app.services.http_client.time.sleep")
    @patch("app.services.http_client.requests.request")
    def test_retry_after_honored_on_503(self, mock_request, mock_sleep):
        mock_request.return_value = self._error_response(503, {"Retry-After": "5"})

        with self.assertRaises(requests.HTTPError):
            request_with_backoff("GET", "https://example.invalid/x", attempts=2, scraper_config=self._FAST_CONFIG)

        mock_sleep.assert_called_once_with(5.0)

    @patch("app.services.http_client.time.sleep")
    @patch("app.services.http_client.requests.request")
    def test_hostile_retry_after_is_clamped(self, mock_request, mock_sleep):
        mock_request.return_value = self._error_response(429, {"Retry-After": "999999"})

        with self.assertRaises(requests.HTTPError):
            request_with_backoff("GET", "https://example.invalid/x", attempts=2, scraper_config=self._FAST_CONFIG)

        mock_sleep.assert_called_once_with(120.0)  # clamped, never an hours-long hang

    @patch("app.services.http_client.time.sleep")
    @patch("app.services.http_client.requests.request")
    def test_malformed_retry_after_falls_back_to_exponential(self, mock_request, mock_sleep):
        mock_request.return_value = self._error_response(429, {"Retry-After": "soon"})

        with self.assertRaises(requests.HTTPError):
            request_with_backoff("GET", "https://example.invalid/x", attempts=2, scraper_config=self._FAST_CONFIG)

        mock_sleep.assert_called_once_with(1.25)


class DataSourcesSettingsRouteTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/settings")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_settings_page_renders_data_sources_and_feed_sources_ui(self):
        with patch.dict(os.environ, _NO_KEY_ENV):
            response = self.client.get("/settings")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="data-sources-form"', html)
        self.assertIn('name="openalex_api_key"', html)
        self.assertIn('name="s2_api_key"', html)
        self.assertIn('name="github_token"', html)
        self.assertIn('id="feed-sources-list"', html)
        self.assertIn('id="feed-category"', html)
        self.assertIn("cs.RO", html)  # category picker built from ARXIV_CATEGORY_NAMES

    def test_save_writes_0600_dotfiles_and_never_echoes_secret(self):
        token = self._csrf_token()
        with patch.dict(os.environ, _NO_KEY_ENV):
            response = self.client.post(
                "/settings/data-sources",
                data={
                    "csrf_token": token,
                    "openalex_api_key": "oa-secret-123",
                    "s2_api_key": "s2-secret-456",
                    "github_token": "gh-secret-789",
                },
            )
            self.assertEqual(response.status_code, 302)

            for source, secret in [
                ("openalex", "oa-secret-123"),
                ("semantic_scholar", "s2-secret-456"),
                ("github", "gh-secret-789"),
            ]:
                path = data_source_key_path(source)
                self.assertEqual(path.read_text(encoding="utf-8"), secret)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

            page = self.client.get("/settings").get_data(as_text=True)
        for secret in ("oa-secret-123", "s2-secret-456", "gh-secret-789"):
            self.assertNotIn(secret, page)  # configured state shown without echoing
        self.assertIn("********", page)
        self.assertIn("Configured", page)

    def test_masked_resubmission_does_not_clobber_existing_keys(self):
        write_secret_file(data_source_key_path("openalex"), "original-key")
        token = self._csrf_token()
        with patch.dict(os.environ, _NO_KEY_ENV):
            response = self.client.post(
                "/settings/data-sources",
                data={"csrf_token": token, "openalex_api_key": "********", "s2_api_key": "", "github_token": ""},
            )
            self.assertEqual(response.status_code, 302)
            self.assertEqual(data_source_key_path("openalex").read_text(encoding="utf-8"), "original-key")
            self.assertFalse(data_source_key_path("semantic_scholar").exists())

    def test_save_requires_csrf_token(self):
        response = self.client.post("/settings/data-sources", data={"openalex_api_key": "x"})
        self.assertEqual(response.status_code, 400)

    def test_env_configured_key_shows_configured_state(self):
        with patch.dict(os.environ, {**_NO_KEY_ENV, "OPENALEX_API_KEY": "env-key"}):
            page = self.client.get("/settings").get_data(as_text=True)
        self.assertIn("Configured", page)
        self.assertNotIn("env-key", page)


class HelpDocsFeedSourceClaimsTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_help_pages_point_to_research_setup_feed_sources(self):
        for page in ("cli", "faq"):
            response = self.client.get(f"/help/{page}")
            self.assertEqual(response.status_code, 200)
            self.assertIn("Research Setup", response.get_data(as_text=True))

    def test_help_settings_page_mentions_data_source_keys(self):
        response = self.client.get("/help/settings")
        self.assertEqual(response.status_code, 200)
        self.assertIn("data-source API keys", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
