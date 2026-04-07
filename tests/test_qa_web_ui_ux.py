from __future__ import annotations

from datetime import date, datetime, timezone

from app.models import Paper, db
from tests.helpers import FlaskDBTestCase


def _make_paper(**overrides) -> Paper:
    today = date.today()
    defaults = dict(
        arxiv_id="2604.09001",
        title="Vision Systems With Multi-Modal Supervision",
        authors="Alice Example, Bob Example",
        link="https://arxiv.org/abs/2604.09001",
        pdf_link="https://arxiv.org/pdf/2604.09001",
        abstract_text="A paper about multimodal computer vision.",
        summary_text="A concise summary for the dashboard card.",
        topic_tags=["Vision", "Multimodal", "Retrieval"],
        categories=["cs.CV"],
        resource_links=[{"type": "code", "url": "https://example.com/code"}],
        match_type="Author + Title",
        matched_terms=["Alice Example", "Vision"],
        paper_score=52.0,
        feedback_score=4,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    defaults.update(overrides)
    return Paper(**defaults)


class WebUiQaTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_empty_dashboard_shows_onboarding_and_theme_toggle(self):
        response = self.client.get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Getting Started", text)
        self.assertIn("Add interests", text)
        self.assertIn("Run a scrape", text)
        self.assertIn('id="theme-toggle"', text)
        self.assertIn("cv-arxiv-theme", text)
        self.assertIn("grid-cols-1 gap-3 lg:grid-cols-3", text)

    def test_dashboard_cards_render_metadata_badges_and_score_details(self):
        db.session.add(_make_paper())
        db.session.commit()

        response = self.client.get("/?timeframe=all")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Vision Systems With Multi-Modal Supervision", text)
        self.assertIn("Alice Example", text)
        self.assertIn("Author</span>", text)
        self.assertIn("Title</span>", text)
        self.assertIn("A concise summary for the dashboard card.", text)
        self.assertIn("Vision", text)
        self.assertIn("Multimodal", text)
        self.assertIn("Score 57.0", text)
        self.assertIn("Why this ranked here", text)

    def test_dashboard_includes_realtime_and_ajax_hooks(self):
        db.session.add(_make_paper())
        db.session.commit()

        response = self.client.get("/?timeframe=all")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("/api/scrape/stream", text)
        self.assertIn("sendFeedback(", text)
        self.assertIn("reading-status", text)
        self.assertIn("notes-textarea", text)
        self.assertIn("bulkExportBibtex()", text)
