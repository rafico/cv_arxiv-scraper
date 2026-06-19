from __future__ import annotations

from app.ui import UI_COOKIE
from tests.helpers import FlaskDBTestCase

# Pages that exist in both the modern and classic UIs.
TOGGLEABLE_PAGES = ("/", "/discover", "/settings", "/help/start")


class UiToggleTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_default_is_modern_ui(self):
        """With no cookie, every page renders the modern sidebar shell."""
        for path in TOGGLEABLE_PAGES:
            text = self.client.get(path).get_data(as_text=True)
            self.assertIn('id="app-sidebar"', text, path)
            self.assertIn("style.css", text, path)
            self.assertNotIn("style.classic.css", text, path)
            self.assertIn("/ui/classic", text, path)  # offers the classic switch

    def test_classic_cookie_renders_classic_ui(self):
        """The classic cookie swaps every page to the vendored classic templates."""
        self.client.set_cookie(UI_COOKIE, "classic")
        for path in TOGGLEABLE_PAGES:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            text = response.get_data(as_text=True)
            self.assertIn("style.classic.css", text, path)
            self.assertNotIn('id="app-sidebar"', text, path)  # no modern shell
            self.assertIn("/ui/modern", text, path)  # offers the switch back

    def test_toggle_sets_and_clears_cookie(self):
        classic = self.client.get("/ui/classic", headers={"Referer": "/settings"})
        self.assertEqual(classic.status_code, 302)
        self.assertEqual(classic.headers["Location"], "/settings")
        self.assertIn(f"{UI_COOKIE}=classic", classic.headers["Set-Cookie"])

        modern = self.client.get("/ui/modern", headers={"Referer": "/"})
        self.assertEqual(modern.status_code, 302)
        # Cleared cookie is sent back with an immediate/empty expiry.
        self.assertIn(f"{UI_COOKIE}=;", modern.headers["Set-Cookie"])

    def test_toggle_ignores_offsite_referrer(self):
        """An external referrer must not become an open redirect target."""
        response = self.client.get("/ui/classic", headers={"Referer": "https://evil.example/x"})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/")
