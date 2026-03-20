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
