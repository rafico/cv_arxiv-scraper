from __future__ import annotations

from pathlib import Path

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
