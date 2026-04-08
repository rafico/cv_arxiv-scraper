from __future__ import annotations

from pathlib import Path

from app.services.mendeley import MendeleyClient
from app.services.zotero import ZoteroClient
from tests.helpers import FlaskDBTestCase


class CsrfSessionQaTests(FlaskDBTestCase):
    def test_dashboard_and_settings_render_csrf_meta_token(self):
        client = self.app.test_client()

        dashboard = client.get("/")
        settings = client.get("/settings")

        self.assertEqual(dashboard.status_code, 200)
        self.assertIn('meta name="csrf-token"', dashboard.get_data(as_text=True))
        self.assertEqual(settings.status_code, 200)
        self.assertIn('meta name="csrf-token"', settings.get_data(as_text=True))

    def test_csrf_token_rotates_across_sessions(self):
        first = self.app.test_client()
        second = self.app.test_client()

        first.get("/")
        with first.session_transaction() as session:
            first_token = session["settings_csrf_token"]

        second.get("/")
        with second.session_transaction() as session:
            second_token = session["settings_csrf_token"]

        self.assertNotEqual(first_token, second_token)

    def test_settings_page_does_not_leak_llm_api_key_contents(self):
        key_path = Path(self.app.config["LLM_KEY_PATH"])
        key_path.write_text("super-secret-openrouter-key", encoding="utf-8")
        self.app.config["SCRAPER_CONFIG"]["llm"] = {
            "enabled": True,
            "provider": "openrouter",
            "model": "openai/gpt-4.1-mini",
            "base_url": "https://openrouter.ai/api/v1",
            "max_concurrent": 2,
        }

        response = self.app.test_client().get("/settings")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("super-secret-openrouter-key", text)
        self.assertIn("********", text)


class CredentialPermissionQaTests(FlaskDBTestCase):
    def test_mendeley_credentials_written_with_restricted_permissions(self):
        credentials_path = Path(self._tmpdir.name) / "mendeley_credentials.json"
        client = MendeleyClient(
            credentials_path=credentials_path, token_path=Path(self._tmpdir.name) / ".mendeley_token"
        )

        client._save_credentials("client-id", "client-secret")

        self.assertEqual(credentials_path.stat().st_mode & 0o777, 0o600)

    def test_zotero_credentials_written_with_restricted_permissions(self):
        credentials_path = Path(self._tmpdir.name) / ".zotero_credentials"
        client = ZoteroClient(credentials_path=credentials_path)

        client._save_credentials("api-key", "user-id")

        self.assertEqual(credentials_path.stat().st_mode & 0o777, 0o600)
