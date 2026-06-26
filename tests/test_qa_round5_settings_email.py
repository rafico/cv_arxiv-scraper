"""QA round 5 regression test — R5-con10 (S2): saving Email settings must validate
the full config before activating it, like the other mutating settings routes.

Pre-fix, _save_config_key wrote + activated the loaded config with no validation, so
an unrelated email-settings save would silently activate a config.yaml that had
drifted to a semantically-invalid (but dict-shaped) state.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

from tests.helpers import TEST_SCRAPER_CONFIG, FlaskDBTestCase


class EmailSaveValidatesConfigTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_email_save_does_not_activate_drifted_invalid_config(self):
        token = self._csrf_token()

        # Drift config.yaml on disk to a dict-but-semantically-invalid state.
        drifted = copy.deepcopy(TEST_SCRAPER_CONFIG)
        drifted["llm"]["provider"] = "invalidprovider"
        Path(self.app.config["CONFIG_PATH"]).write_text(yaml.safe_dump(drifted), encoding="utf-8")

        response = self.client.post(
            "/settings/email",
            data={"email_recipient": "me@example.com", "email_subject_prefix": "Digest"},
            headers={"X-CSRF-Token": token},
        )

        # The route handles it (no 500) but must NOT activate the invalid config.
        self.assertIn(response.status_code, (200, 302))
        self.assertEqual(self.app.config["SCRAPER_CONFIG"]["llm"]["provider"], "openrouter")


if __name__ == "__main__":
    unittest.main()
