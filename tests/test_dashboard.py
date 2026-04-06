import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

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

        static_root = Path(self._tmpdir.name) / "static"
        static_root.mkdir(parents=True, exist_ok=True)
        self.app.static_folder = str(static_root)
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

    def test_dashboard_uses_app_thumbnail_route(self):
        paper = Paper.query.filter_by(title="Paper 0").one()

        response = self.client.get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(f'/papers/{paper.id}/thumbnail.png', text)
        self.assertNotIn("cdn-thumbnails.huggingface.co", text)

    @patch("app.services.mendeley.MendeleyClient.check_connection")
    def test_dashboard_shows_mendeley_button_when_connected(self, mock_check_connection):
        mock_check_connection.return_value = {"status": "connected", "message": "Mendeley is connected."}

        response = self.client.get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Send to Mendeley", text)
        self.assertIn("mendeley-sync-btn", text)

    @patch("app.services.mendeley.MendeleyClient.check_connection")
    def test_dashboard_hides_mendeley_button_when_not_connected(self, mock_check_connection):
        mock_check_connection.return_value = {"status": "no_token", "message": "Mendeley not authorized."}

        response = self.client.get("/")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Send to Mendeley", text)
        self.assertNotIn("mendeley-sync-btn", text)

    def test_paper_thumbnail_route_serves_existing_thumbnail(self):
        paper = Paper.query.filter_by(title="Paper 0").one()
        thumbnails_dir = Path(self.app.static_folder) / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_path = thumbnails_dir / f"{paper.arxiv_id}.png"
        thumbnail_path.write_bytes(b"png-bytes")

        response = self.client.get(f"/papers/{paper.id}/thumbnail.png")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")
        self.assertEqual(response.get_data(), b"png-bytes")

    def test_paper_thumbnail_route_generates_missing_thumbnail(self):
        paper = Paper.query.filter_by(title="Paper 0").one()

        def _fake_generate_thumbnail(arxiv_id, pdf_link, static_root, session=None, pdf_content=None):
            thumbnail_path = Path(static_root) / "thumbnails" / f"{arxiv_id}.png"
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
            thumbnail_path.write_bytes(b"generated-png")
            return True

        with patch("app.routes.dashboard.generate_thumbnail", side_effect=_fake_generate_thumbnail) as mock_generate:
            response = self.client.get(f"/papers/{paper.id}/thumbnail.png")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")
        self.assertEqual(response.get_data(), b"generated-png")
        mock_generate.assert_called_once_with(paper.arxiv_id, paper.pdf_link, Path(self.app.static_folder))

    def test_paper_thumbnail_route_returns_404_when_thumbnail_cannot_be_generated(self):
        paper = Paper.query.filter_by(title="Paper 0").one()

        with patch("app.routes.dashboard.generate_thumbnail", return_value=False) as mock_generate:
            response = self.client.get(f"/papers/{paper.id}/thumbnail.png")

        self.assertEqual(response.status_code, 404)
        mock_generate.assert_called_once_with(paper.arxiv_id, paper.pdf_link, Path(self.app.static_folder))
