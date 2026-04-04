from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from app.models import Collection, Paper, PaperCollection, db
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


class CorpusApiTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        seed_paper = Paper(
            arxiv_id="2603.0101",
            title="Seed Paper",
            authors="Author A",
            link="https://arxiv.org/abs/2603.0101",
            pdf_link="https://arxiv.org/pdf/2603.0101",
            abstract_text="seed abstract",
            match_type="Title",
            matched_terms=["Vision"],
            paper_score=1.0,
            publication_date="2026-03-19",
            scraped_date="2026-03-19",
        )
        db.session.add(seed_paper)
        db.session.commit()
        self.seed_paper = seed_paper

    def test_corpus_clusters_endpoint_forwards_query_arguments(self):
        with patch("app.services.corpus_analysis.analyze_topic_clusters") as mock_analyze:
            mock_analyze.return_value = {"window_days": 14, "offset_days": 2, "clusters": []}

            response = self.client.get(
                "/api/corpus/clusters?window_days=14&offset_days=2&clusters=3&limit=150&paper_limit=4"
            )

        self.assertEqual(response.status_code, 200)
        mock_analyze.assert_called_once_with(
            window_days=14,
            offset_days=2,
            limit=150,
            cluster_count=3,
            paper_limit=4,
        )
        self.assertEqual(response.get_json()["window_days"], 14)

    def test_corpus_emerging_endpoint_rejects_invalid_days(self):
        response = self.client.get("/api/corpus/emerging?recent_days=abc")

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_corpus_neighbors_endpoint_uses_collection_and_whitelist_authors(self):
        collection = Collection(name="Saved Set", description="", color="#123456")
        db.session.add(collection)
        db.session.commit()
        db.session.add(PaperCollection(collection_id=collection.id, paper_id=self.seed_paper.id))
        db.session.commit()
        self.app.config["SCRAPER_CONFIG"]["whitelists"]["authors"] = ["Tracked Author"]

        with patch("app.services.corpus_analysis.find_neighbor_papers") as mock_neighbors:
            mock_neighbors.return_value = {"seed_paper_ids": [self.seed_paper.id], "results": []}

            response = self.client.get(
                f"/api/corpus/neighbors?collection_id={collection.id}&limit=7&exclude_tracked_authors=0"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["collection_id"], collection.id)
        self.assertEqual(mock_neighbors.call_args.args[0], [self.seed_paper.id])
        self.assertEqual(mock_neighbors.call_args.kwargs["limit"], 7)
        self.assertEqual(mock_neighbors.call_args.kwargs["tracked_authors"], ["Tracked Author"])
        self.assertFalse(mock_neighbors.call_args.kwargs["exclude_tracked_authors"])
