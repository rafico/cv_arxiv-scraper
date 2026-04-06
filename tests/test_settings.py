from __future__ import annotations

import io
import os
from pathlib import Path
from unittest.mock import patch

import yaml

from tests.helpers import FlaskDBTestCase


class SettingsRouteTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _payload(self, csrf_token: str = "") -> dict[str, str]:
        return {
            "csrf_token": csrf_token,
            "titles": "Vision\nSegmentation",
            "authors": "Jane Doe\nJohn Smith",
            "affiliations": "MIT\nStanford",
        }

    def _csrf_token(self) -> str:
        self.client.get("/settings")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_settings_post_requires_valid_csrf_token(self):
        response = self.client.post("/settings", data=self._payload())
        self.assertEqual(response.status_code, 400)

    def test_settings_post_saves_with_valid_csrf_token(self):
        token = self._csrf_token()
        response = self.client.post("/settings", data=self._payload(token))

        self.assertEqual(response.status_code, 302)
        self.assertIn("/settings", response.headers["Location"])

        expected = {
            "titles": ["Vision", "Segmentation"],
            "authors": ["Jane Doe", "John Smith"],
            "affiliations": ["MIT", "Stanford"],
        }
        self.assertEqual(self.app.config["SCRAPER_CONFIG"]["whitelists"], expected)

        config_path = Path(self.app.config["CONFIG_PATH"])
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["whitelists"], expected)

    def test_settings_page_shows_callback_uri(self):
        """The no_credentials state shows the callback URI for setup."""
        with patch("app.services.email_digest.DEFAULT_CREDENTIALS_PATH") as mock_creds:
            mock_creds.exists.return_value = False
            response = self.client.get("/settings")
        self.assertIn(b"gmail-callback", response.data)
        self.assertIn(b"mendeley-callback", response.data)


class GmailStatusTests(FlaskDBTestCase):
    """Tests for /settings/gmail-status endpoint."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    @patch("app.services.email_digest.DEFAULT_TOKEN_PATH")
    @patch("app.services.email_digest.DEFAULT_CREDENTIALS_PATH")
    def test_gmail_status_no_token(self, mock_creds, mock_token):
        mock_creds.exists.return_value = True
        mock_token.exists.return_value = False

        response = self.client.get("/settings/gmail-status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "no_token")

    @patch("app.services.email_digest.DEFAULT_CREDENTIALS_PATH")
    def test_gmail_status_no_credentials(self, mock_creds):
        mock_creds.exists.return_value = False

        response = self.client.get("/settings/gmail-status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "no_credentials")


class UploadCredentialsTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/settings")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_upload_requires_csrf(self):
        response = self.client.post("/settings/upload-credentials")
        self.assertEqual(response.status_code, 400)

    @patch("app.services.email_digest.DEFAULT_CREDENTIALS_PATH")
    def test_upload_saves_valid_json(self, mock_creds_path):
        token = self._csrf_token()
        valid_json = b'{"web":{"client_id":"123","client_secret":"456"}}'

        response = self.client.post(
            "/settings/upload-credentials",
            data={
                "csrf_token": token,
                "credentials_file": (io.BytesIO(valid_json), "credentials.json"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/settings", response.headers["Location"])
        mock_creds_path.write_bytes.assert_called_once_with(valid_json)

    @patch("app.services.email_digest.DEFAULT_CREDENTIALS_PATH")
    def test_upload_rejects_invalid_json(self, mock_creds_path):
        token = self._csrf_token()
        invalid_json = b"Not a JSON"

        response = self.client.post(
            "/settings/upload-credentials",
            data={
                "csrf_token": token,
                "credentials_file": (io.BytesIO(invalid_json), "credentials.json"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 302)
        mock_creds_path.write_bytes.assert_not_called()

    @patch("app.services.email_digest.DEFAULT_CREDENTIALS_PATH")
    def test_upload_rejects_missing_oauth_fields(self, mock_creds_path):
        token = self._csrf_token()
        wrong_json = b'{"installed":{"client_id":"123"}}'

        response = self.client.post(
            "/settings/upload-credentials",
            data={
                "csrf_token": token,
                "credentials_file": (io.BytesIO(wrong_json), "credentials.json"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 302)
        mock_creds_path.write_bytes.assert_not_called()


class MendeleySettingsTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/settings")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    @patch("app.services.mendeley.MendeleyClient._save_credentials")
    def test_mendeley_setup_saves_direct_credentials(self, mock_save):
        token = self._csrf_token()

        response = self.client.post(
            "/settings/mendeley-setup",
            data={
                "csrf_token": token,
                "mendeley_client_id": "client-123",
                "mendeley_client_secret": "secret-456",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/settings", response.headers["Location"])
        mock_save.assert_called_once_with("client-123", "secret-456")

    @patch("app.services.mendeley.MendeleyClient._save_credentials")
    def test_mendeley_setup_rejects_missing_fields(self, mock_save):
        token = self._csrf_token()

        response = self.client.post(
            "/settings/mendeley-setup",
            data={
                "csrf_token": token,
                "mendeley_client_id": "client-123",
                "mendeley_client_secret": "",
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_save.assert_not_called()

    @patch("app.services.mendeley.MendeleyClient._save_credentials")
    def test_upload_mendeley_credentials_saves_parsed_json(self, mock_save):
        token = self._csrf_token()
        valid_json = b'{"client_id":"123","client_secret":"456"}'

        response = self.client.post(
            "/settings/upload-mendeley-credentials",
            data={
                "csrf_token": token,
                "mendeley_credentials_file": (io.BytesIO(valid_json), "mendeley_credentials.json"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/settings", response.headers["Location"])
        mock_save.assert_called_once_with("123", "456")


class GmailOAuthFlowTests(FlaskDBTestCase):
    """Tests for the Gmail OAuth redirect flow."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/settings")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    @patch("app.services.email_digest.start_oauth_flow")
    def test_gmail_auth_redirects_to_google(self, mock_start):
        mock_start.return_value = {
            "success": True,
            "auth_url": "https://accounts.google.com/o/oauth2/auth?state=abc123",
            "state": "abc123",
            "message": "Redirecting.",
        }
        token = self._csrf_token()
        response = self.client.post("/settings/gmail-auth", data={"csrf_token": token})

        self.assertEqual(response.status_code, 302)
        self.assertIn("accounts.google.com", response.headers["Location"])

        with self.client.session_transaction() as sess:
            self.assertEqual(sess["oauth_state"], "abc123")

    @patch("app.services.email_digest.start_oauth_flow")
    def test_gmail_auth_shows_error_on_failure(self, mock_start):
        mock_start.return_value = {
            "success": False,
            "auth_url": None,
            "state": None,
            "message": "credentials.json not found.",
        }
        token = self._csrf_token()
        response = self.client.post("/settings/gmail-auth", data={"csrf_token": token})

        self.assertEqual(response.status_code, 302)
        self.assertIn("/settings", response.headers["Location"])

    def test_gmail_callback_rejects_mismatched_state(self):
        with self.client.session_transaction() as sess:
            sess["oauth_state"] = "expected_state"

        response = self.client.get("/settings/gmail-callback?state=wrong&code=abc")
        self.assertEqual(response.status_code, 302)

    @patch("app.services.email_digest.finish_oauth_flow")
    def test_gmail_callback_success(self, mock_finish):
        mock_finish.return_value = {
            "success": True,
            "message": "Gmail authorized successfully.",
        }
        with self.client.session_transaction() as sess:
            sess["oauth_state"] = "good_state"

        response = self.client.get("/settings/gmail-callback?state=good_state&code=authcode")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/settings", response.headers["Location"])
        mock_finish.assert_called_once()

    def test_gmail_callback_handles_google_error(self):
        with self.client.session_transaction() as sess:
            sess["oauth_state"] = "some_state"

        response = self.client.get("/settings/gmail-callback?state=some_state&error=access_denied")
        self.assertEqual(response.status_code, 302)


class EmailSettingsTests(FlaskDBTestCase):
    """Tests for /settings/email endpoint."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/settings")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_email_settings_save(self):
        token = self._csrf_token()
        response = self.client.post(
            "/settings/email",
            data={
                "csrf_token": token,
                "email_recipient": "test@example.com",
                "email_subject_prefix": "My Digest",
            },
        )

        self.assertEqual(response.status_code, 302)

        email_cfg = self.app.config["SCRAPER_CONFIG"]["email"]
        self.assertEqual(email_cfg["recipient"], "test@example.com")
        self.assertEqual(email_cfg["subject_prefix"], "My Digest")

        config_path = Path(self.app.config["CONFIG_PATH"])
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["email"]["recipient"], "test@example.com")

    def test_email_settings_requires_csrf(self):
        response = self.client.post(
            "/settings/email",
            data={
                "email_recipient": "test@example.com",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_settings_page_shows_email_config(self):
        self.app.config["SCRAPER_CONFIG"]["email"] = {
            "recipient": "me@test.com",
            "subject_prefix": "Test Prefix",
        }
        response = self.client.get("/settings")
        self.assertEqual(response.status_code, 200)
        html = response.data.decode()
        self.assertIn("me@test.com", html)
        self.assertIn("Test Prefix", html)
        self.assertIn("Email &amp; Gmail", html)

    def test_digest_preview_route_renders_html(self):
        response = self.client.get("/settings/digest-preview")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ArXiv CV Digest", response.get_data(as_text=True))


class PreferenceSettingsTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/settings")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_preferences_save_updates_config(self):
        token = self._csrf_token()
        response = self.client.post(
            "/settings/preferences",
            data={
                "csrf_token": token,
                "pref_author_weight": "50",
                "pref_affiliation_weight": "20",
                "pref_title_weight": "10",
                "pref_ai_weight": "3",
                "pref_freshness_half_life_days": "7",
                "muted_topics": "Tracking\nDetection",
                "muted_authors": "Alice Smith",
                "muted_affiliations": "Example Lab",
            },
        )

        self.assertEqual(response.status_code, 302)
        preferences = self.app.config["SCRAPER_CONFIG"]["preferences"]
        self.assertEqual(preferences["ranking"]["author_weight"], 50.0)
        self.assertEqual(preferences["ranking"]["freshness_half_life_days"], 7.0)
        self.assertEqual(preferences["muted"]["topics"], ["Tracking", "Detection"])
        self.assertEqual(preferences["muted"]["authors"], ["Alice Smith"])


class LLMSettingsTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/settings")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_settings_page_shows_llm_section(self):
        response = self.client.get("/settings")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("LLM / AI", html)

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False)
    def test_llm_settings_save_writes_key_file_not_config(self):
        token = self._csrf_token()
        response = self.client.post(
            "/settings/llm",
            data={
                "csrf_token": token,
                "llm_enabled": "on",
                "llm_provider": "openrouter",
                "llm_api_key": "super-secret-key",
                "llm_model": "test/model",
                "llm_base_url": "https://openrouter.ai/api/v1",
                "llm_max_concurrent": "3",
            },
        )

        self.assertEqual(response.status_code, 302)

        config_path = Path(self.app.config["CONFIG_PATH"])
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["llm"]["model"], "test/model")
        self.assertEqual(saved["llm"]["provider"], "openrouter")
        self.assertNotIn("api_key", saved["llm"])

        key_path = Path(self.app.config["LLM_KEY_PATH"])
        self.assertTrue(key_path.exists())
        self.assertEqual(key_path.read_text(encoding="utf-8"), "super-secret-key")

    def test_llm_settings_save_ollama_skips_api_key_write(self):
        token = self._csrf_token()
        response = self.client.post(
            "/settings/llm",
            data={
                "csrf_token": token,
                "llm_enabled": "on",
                "llm_provider": "ollama",
                "llm_model": "mistral",
                "llm_base_url": "http://localhost:11434/v1",
                "llm_max_concurrent": "2",
            },
        )

        self.assertEqual(response.status_code, 302)

        config_path = Path(self.app.config["CONFIG_PATH"])
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["llm"]["provider"], "ollama")
        self.assertEqual(saved["llm"]["model"], "mistral")
        self.assertEqual(saved["llm"]["base_url"], "http://localhost:11434/v1")
        self.assertTrue(saved["llm"]["enabled"])

        key_path = Path(self.app.config["LLM_KEY_PATH"])
        self.assertFalse(key_path.exists())

    def test_llm_settings_save_ollama_applies_defaults(self):
        token = self._csrf_token()
        response = self.client.post(
            "/settings/llm",
            data={
                "csrf_token": token,
                "llm_provider": "ollama",
                "llm_model": "",
                "llm_base_url": "",
                "llm_max_concurrent": "4",
            },
        )

        self.assertEqual(response.status_code, 302)

        config_path = Path(self.app.config["CONFIG_PATH"])
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["llm"]["model"], "gemma3")
        self.assertEqual(saved["llm"]["base_url"], "http://localhost:11434/v1")

    def test_settings_page_shows_provider_select(self):
        response = self.client.get("/settings")
        html = response.get_data(as_text=True)
        self.assertIn("llm_provider", html)
        self.assertIn("ollama", html)

    def test_settings_page_uses_ollama_defaults_when_model_and_base_url_missing(self):
        self.app.config["SCRAPER_CONFIG"]["llm"] = {
            "enabled": False,
            "provider": "ollama",
            "max_concurrent": 4,
        }

        response = self.client.get("/settings")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('value="gemma3"', html)
        self.assertIn('value="http://localhost:11434/v1"', html)
