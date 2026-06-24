from __future__ import annotations

from app.services.email_digest import _get_email_config, build_digest_preview
from tests.helpers import FlaskDBTestCase


class EmailConfigMalformedSectionTests(FlaskDBTestCase):
    """Regression tests for G7: a malformed 'email' config section (YAML null or
    a scalar string) must not crash with AttributeError->500."""

    def _set_email_section(self, value) -> None:
        self.app.config["SCRAPER_CONFIG"]["email"] = value

    # ── _get_email_config ────────────────────────────────────────────────

    def test_g7_get_email_config_with_null_section_does_not_raise(self):
        self._set_email_section(None)
        cfg = _get_email_config(self.app)
        self.assertEqual(cfg["recipient"], "")
        self.assertEqual(cfg["subject_prefix"], "ArXiv Digest")

    def test_g7_get_email_config_with_scalar_section_does_not_raise(self):
        self._set_email_section("me@example.com")
        cfg = _get_email_config(self.app)
        self.assertEqual(cfg["recipient"], "")
        self.assertEqual(cfg["subject_prefix"], "ArXiv Digest")

    # ── build_digest_preview ─────────────────────────────────────────────

    def test_g7_build_digest_preview_with_null_section_does_not_raise(self):
        self._set_email_section(None)
        preview = build_digest_preview(self.app)
        self.assertEqual(preview["recipient"], "")

    def test_g7_build_digest_preview_with_scalar_section_does_not_raise(self):
        self._set_email_section("me@example.com")
        preview = build_digest_preview(self.app)
        self.assertEqual(preview["recipient"], "")

    # ── GET /settings ────────────────────────────────────────────────────

    def test_g7_settings_page_200_with_null_email_section(self):
        self._set_email_section(None)
        client = self.app.test_client()
        resp = client.get("/settings")
        self.assertEqual(resp.status_code, 200)

    def test_g7_settings_page_200_with_scalar_email_section(self):
        self._set_email_section("me@example.com")
        client = self.app.test_client()
        resp = client.get("/settings")
        self.assertEqual(resp.status_code, 200)
