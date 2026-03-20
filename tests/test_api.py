from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from app.models import Paper, db
from tests.helpers import FlaskDBTestCase


class ApiCsrfTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        db.session.add(
            Paper(
                arxiv_id="2603.0001",
                title="API Paper",
                authors="Author A, Author B",
                link="https://arxiv.org/abs/2603.0001",
                pdf_link="https://arxiv.org/pdf/2603.0001",
                topic_tags=["Segmentation"],
                match_type="Title",
                matched_terms=["Vision"],
                paper_score=1.0,
                publication_date="2026-03-19",
                scraped_date="2026-03-19",
            )
        )
        db.session.commit()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_feedback_endpoint_requires_csrf(self):
        paper = Paper.query.first()
        response = self.client.post(
            f"/api/papers/{paper.id}/feedback",
            json={"action": "save"},
        )
        self.assertEqual(response.status_code, 400)

    @patch("app.routes.api.SCRAPE_JOB_MANAGER.start_or_get_active")
    def test_scrape_endpoint_requires_csrf(self, mock_start):
        response = self.client.post("/api/scrape", json={})
        self.assertEqual(response.status_code, 400)
        mock_start.assert_not_called()

    @patch("app.routes.api.SCRAPE_JOB_MANAGER.start_or_get_active")
    def test_scrape_endpoint_accepts_csrf_header(self, mock_start):
        mock_start.return_value = SimpleNamespace(
            id="job-1",
            status="running",
            started_at=datetime(2026, 3, 19, 10, 0, 0),
        )

        response = self.client.post(
            "/api/scrape",
            json={},
            headers={"X-CSRF-Token": self._csrf_token()},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["job_id"], "job-1")

    def test_scrape_stream_requires_csrf(self):
        response = self.client.get("/api/scrape/stream")
        self.assertEqual(response.status_code, 400)

    def test_follow_endpoint_adds_author_to_whitelist(self):
        paper = Paper.query.first()
        response = self.client.post(
            f"/api/papers/{paper.id}/follow",
            json={},
            headers={"X-CSRF-Token": self._csrf_token()},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Author A", self.app.config["SCRAPER_CONFIG"]["whitelists"]["authors"])

    def test_mute_endpoint_adds_topic_to_preferences(self):
        paper = Paper.query.first()
        response = self.client.post(
            f"/api/papers/{paper.id}/mute",
            json={},
            headers={"X-CSRF-Token": self._csrf_token()},
        )

        self.assertEqual(response.status_code, 200)
        muted_topics = self.app.config["SCRAPER_CONFIG"]["preferences"]["muted"]["topics"]
        self.assertIn("Segmentation", muted_topics)
