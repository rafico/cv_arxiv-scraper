from __future__ import annotations

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

        response = self.client.get(
            "/settings/gmail-callback?state=some_state&error=access_denied"
        )
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
        response = self.client.post("/settings/email", data={
            "csrf_token": token,
            "email_recipient": "test@example.com",
            "email_subject_prefix": "My Digest",
        })

        self.assertEqual(response.status_code, 302)

        email_cfg = self.app.config["SCRAPER_CONFIG"]["email"]
        self.assertEqual(email_cfg["recipient"], "test@example.com")
        self.assertEqual(email_cfg["subject_prefix"], "My Digest")

        config_path = Path(self.app.config["CONFIG_PATH"])
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["email"]["recipient"], "test@example.com")

    def test_email_settings_requires_csrf(self):
        response = self.client.post("/settings/email", data={
            "email_recipient": "test@example.com",
        })
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
        self.assertIn("Email & Gmail", html)

