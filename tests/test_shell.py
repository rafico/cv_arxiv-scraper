from __future__ import annotations

import re

from app.enums import FeedbackAction
from app.models import Collection, PaperFeedback, SavedSearch, db
from tests.helpers import FlaskDBTestCase


class ShellTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_sidebar_renders_on_every_page(self):
        for path in ("/", "/discover", "/settings", "/help/start"):
            response = self.client.get(path)
            text = response.get_data(as_text=True)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn('id="app-sidebar"', text, path)
            self.assertIn('id="theme-toggle"', text, path)
            self.assertIn("Inbox", text, path)
            self.assertIn("Saved", text, path)

    def test_sidebar_shows_saved_count(self):
        db.session.add(PaperFeedback(paper_id=1, action=FeedbackAction.SAVE.value))
        db.session.commit()
        response = self.client.get("/")
        self.assertIn("Saved", response.get_data(as_text=True))

    def test_sidebar_lists_collections_and_saved_searches(self):
        db.session.add(Collection(name="Diffusion"))
        db.session.add(SavedSearch(name="Recent NeRF", filters={"q": "nerf"}))
        db.session.commit()
        text = self.client.get("/").get_data(as_text=True)
        self.assertIn("Diffusion", text)
        self.assertIn("Recent NeRF", text)
        self.assertIn("Collections", text)
        self.assertIn("Saved searches", text)

    def test_new_collection_entry_points_render_without_collections(self):
        # Zero collections must still offer a way to create the first one.
        text = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="new-collection-btn"', text)
        self.assertIn("function newCollection(", text)

    def test_new_collection_works_from_pages_without_their_own_csrf_token(self):
        # The sidebar "+" renders everywhere, so every page must carry the CSRF meta tag.
        for path in ("/graph", "/help/organization"):
            text = self.client.get(path).get_data(as_text=True)
            token = re.search(r'<meta name="csrf-token" content="([^"]+)"', text)
            self.assertIsNotNone(token, path)
            res = self.client.post("/api/collections", json={"name": path}, headers={"X-CSRF-Token": token.group(1)})
            self.assertEqual(res.status_code, 201, path)
