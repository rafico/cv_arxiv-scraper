from __future__ import annotations

from app.models import db
from tests.helpers import FlaskDBTestCase
from tests.test_new_features import _make_paper


class PaperManagementDashboardTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_dashboard_renders_user_notes_tags_and_selected_reading_status(self):
        db.session.add(
            _make_paper(
                user_notes="Important note for later review",
                user_tags=["thesis-ch3", "baseline"],
                reading_status="reading",
            )
        )
        db.session.commit()

        response = self.client.get("/?timeframe=all")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Important note for later review", text)
        self.assertIn("thesis-ch3", text)
        self.assertIn("baseline", text)
        self.assertIn('<option value="reading" selected>', text)
