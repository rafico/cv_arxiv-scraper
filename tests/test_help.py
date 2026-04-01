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
        pages = ["start", "ui", "features", "settings", "faq"]
        for page in pages:
            response = self.client.get(f"/help/{page}")
            self.assertEqual(response.status_code, 200, f"Page /help/{page} failed to load")

    def test_help_content_not_lost_in_split(self):
        # Concatenate text from all pages to ensure no content was lost during the refactor
        pages = ["start", "ui", "features", "settings", "faq"]
        full_text = ""
        for page in pages:
            response = self.client.get(f"/help/{page}")
            full_text += response.get_data(as_text=True)

        self.assertIn("Start Here", full_text)
        self.assertIn("Settings, Explained Simply", full_text)
        self.assertIn("Not Interested", full_text)
        self.assertIn("/settings?section=interests", full_text)

        # New section headings
        for section in [
            "Paper Organization",
            "Reference Manager Sync",
            "Smart Features",
            "Power User Tips",
        ]:
            self.assertIn(section, full_text, f"Missing help section: {section}")

        # Key feature keywords
        for feature in [
            "Collections",
            "User Tags",
            "Reading Status",
            "BibTeX",
            "Mendeley",
            "Zotero",
            "Citation Counts",
            "Follow Author",
            "Mute Topic",
            "Duplicate Detection",
            "Historical Search",
            "Scheduled Scrapes",
            "Keyboard Shortcuts",
            "Advanced Filters",
            "Saved Searches",
            "Bulk Operations",
        ]:
            self.assertIn(feature, full_text, f"Missing feature in help: {feature}")
