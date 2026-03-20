from tests.helpers import FlaskDBTestCase


class HelpRouteTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_help_page_renders_plain_language_guidance(self):
        response = self.client.get("/help")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Start Here", text)
        self.assertIn("Settings, Explained Simply", text)
        self.assertIn("Not Interested", text)
        self.assertIn("/settings?section=interests", text)

    def test_help_page_covers_new_features(self):
        response = self.client.get("/help")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)

        # New section headings
        for section in [
            "Paper Organization",
            "Reference Manager Sync",
            "Smart Features",
            "Power User Tips",
        ]:
            self.assertIn(section, text, f"Missing help section: {section}")

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
            self.assertIn(feature, text, f"Missing feature in help: {feature}")

