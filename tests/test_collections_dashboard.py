from __future__ import annotations

from app.models import Collection, ScrapeRun, db
from tests.helpers import FlaskDBTestCase


class CollectionDashboardTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_empty_collection_filter_shows_empty_state_message(self):
        collection = Collection(name="Empty Collection")
        db.session.add(collection)
        db.session.add(ScrapeRun(status="success"))
        db.session.commit()

        response = self.client.get(f"/?collection={collection.id}&timeframe=all")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No papers for this filter", response.data)
