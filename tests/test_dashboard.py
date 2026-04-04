import re
from datetime import date, datetime, timedelta, timezone

from app.models import Paper, db
from app.services.feedback import apply_feedback_action
from tests.helpers import FlaskDBTestCase


class DashboardRouteTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        today = date.today()

        for idx in range(30):
            publication_dt = today if idx < 20 else today - timedelta(days=45)
            paper = Paper(
                arxiv_id=f"2602.{1000 + idx}",
                title=f"Paper {idx}",
                authors="Author A",
                link=f"https://arxiv.org/abs/2602.{1000 + idx}",
                pdf_link=f"https://arxiv.org/pdf/2602.{1000 + idx}",
                abstract_text="vision transformer segmentation",
                summary_text="Summary text",
                topic_tags=["Segmentation", "Vision"],
                categories=["cs.CV"] if idx % 2 == 0 else ["cs.RO"],
                resource_links=([{"type": "code", "url": f"https://example.com/code/{idx}"}] if idx % 3 == 0 else []),
                match_type="Title",
                matched_terms=["vision"],
                paper_score=10.0 + idx,
                feedback_score=0,
                is_hidden=False,
                publication_date=publication_dt.isoformat(),
                publication_dt=publication_dt,
                scraped_date=today.isoformat(),
                scraped_at=now,
            )
            db.session.add(paper)
        db.session.commit()

        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_default_daily_timeframe_filters_old_papers(self):
        response = self.client.get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Inbox", text)
        self.assertNotIn("Paper 29", text)

    def test_all_time_second_page_available(self):
        response = self.client.get("/?timeframe=all&page=2")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Page 2", text)

    def test_feedback_endpoint_toggles_action(self):
        paper = Paper.query.first()
        token = self._csrf_token()
        response = self.client.post(
            f"/api/papers/{paper.id}/feedback",
            json={"action": "save"},
            headers={"X-CSRF-Token": token},
        )
        data = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["active"])
        self.assertEqual(data["counts"]["save"], 1)

    def test_saved_view_lists_only_saved_papers(self):
        saved_paper = Paper.query.filter_by(title="Paper 0").first()
        apply_feedback_action(saved_paper.id, "save")

        response = self.client.get("/?view=saved")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Saved", text)
        self.assertIn("Paper 0", text)
        self.assertIn("Recently Saved", text)

    def test_category_filter_limits_results(self):
        response = self.client.get("/?timeframe=all&category=cs.RO")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("https://arxiv.org/abs/2602.1001", text)
        self.assertNotIn("https://arxiv.org/abs/2602.1000", text)
        self.assertIn("cs.RO", text)

    def test_resource_filter_limits_results(self):
        response = self.client.get("/?timeframe=all&resource_filter=available")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("https://example.com/code/0", text)
        self.assertNotIn("https://arxiv.org/abs/2602.1001", text)
        self.assertIn("Has resources", text)

    def test_dashboard_shows_citation_provenance_tooltip_and_openalex_fallback(self):
        paper = Paper.query.filter_by(title="Paper 0").one()
        paper.citation_count = None
        paper.openalex_id = "W999"
        paper.openalex_cited_by_count = 17
        paper.citation_source = "openalex"
        paper.citation_provenance = {"source": "openalex", "updated_at": "2026-04-01T12:00:00"}
        db.session.commit()

        response = self.client.get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Citation count from OpenAlex, updated 2026-04-01", text)
        self.assertIn("https://openalex.org/W999", text)
        self.assertRegex(text, re.compile(r">\s*17\s*</a>"))
