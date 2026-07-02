"""Discover page — corpus insights section (topic clusters + emerging topics)."""

from __future__ import annotations

from tests.helpers import FlaskDBTestCase


class DiscoverCorpusInsightsTests(FlaskDBTestCase):
    """The Discover page surfaces the corpus analytics backend as lazy panels."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _page(self) -> str:
        response = self.client.get("/discover")
        self.assertEqual(response.status_code, 200)
        return response.get_data(as_text=True)

    def test_renders_corpus_insights_section(self):
        text = self._page()
        self.assertIn('id="corpus-insights"', text)
        self.assertIn("Corpus insights", text)

    def test_renders_both_collapsible_panels_with_bodies(self):
        text = self._page()
        self.assertIn('id="clusters-panel"', text)
        self.assertIn('id="clusters-body"', text)
        self.assertIn('id="emerging-panel"', text)
        self.assertIn('id="emerging-body"', text)

    def test_panels_are_lazy_details_elements(self):
        # Collapsed <details> without an `open` attribute — the fetch only fires on expand.
        text = self._page()
        self.assertIn('<details id="clusters-panel"', text)
        self.assertIn('<details id="emerging-panel"', text)
        self.assertNotIn('<details id="clusters-panel" open', text)
        self.assertNotIn('<details id="emerging-panel" open', text)

    def test_references_corpus_api_endpoints(self):
        text = self._page()
        self.assertIn("/api/corpus/clusters", text)
        self.assertIn("/api/corpus/emerging", text)
