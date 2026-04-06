from tests.helpers import FlaskDBTestCase


class HelpRouteTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_help_redirects_to_start(self):
        response = self.client.get("/help")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.location.endswith("/help/start"))

    def test_help_pages_load_successfully(self):
        pages = ["start", "ui", "search", "organization", "features", "export", "cli", "settings", "faq"]
        for page in pages:
            response = self.client.get(f"/help/{page}")
            self.assertEqual(response.status_code, 200, f"Page /help/{page} failed to load")

    def test_help_unknown_page_returns_404(self):
        response = self.client.get("/help/nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_help_content_across_pages(self):
        pages = ["start", "ui", "search", "organization", "features", "export", "cli", "settings", "faq"]
        full_text = ""
        for page in pages:
            response = self.client.get(f"/help/{page}")
            full_text += response.get_data(as_text=True)

        # Core navigation and setup
        self.assertIn("Start Here", full_text)
        self.assertIn("Not Interested", full_text)
        self.assertIn("/settings?section=interests", full_text)

        # Section headings across pages
        for section in [
            "Collections",
            "Reference Manager Sync",
            "How Ranking Works",
            "Keyboard Shortcuts",
            "Enrichment Backfills",
            "AI Summaries",
            "Email Digest",
        ]:
            self.assertIn(section, full_text, f"Missing help section: {section}")

        # Key features across all pages
        for feature in [
            "BibTeX",
            "Mendeley",
            "Zotero",
            "Citations",
            "Follow",
            "Mute",
            "Duplicate Detection",
            "Scheduled Scrapes",
            "Saved Searches",
            "Bulk",
            "OpenAlex",
            "SPECTER2",
            "Reading Status",
            "Tags",
        ]:
            self.assertIn(feature, full_text, f"Missing feature in help: {feature}")
