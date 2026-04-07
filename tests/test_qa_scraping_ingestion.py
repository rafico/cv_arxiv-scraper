from __future__ import annotations

import io
import unittest
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from app.models import Paper, ScrapeRun, db
from app.services.jobs import ScrapeJobManager
from tests.helpers import FlaskDBTestCase


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        arxiv_id="2604.00001",
        title="CLI Matched Paper",
        authors="Alice Example, Bob Example",
        link="https://arxiv.org/abs/2604.00001",
        pdf_link="https://arxiv.org/pdf/2604.00001",
        abstract_text="A paper about ingestion QA.",
        summary_text="A concise QA summary.",
        topic_tags=["vision", "retrieval"],
        categories=["cs.CV"],
        match_type="Author + Title",
        matched_terms=["Alice Example", "Vision"],
        paper_score=17.5,
        publication_date="2026-04-07",
        publication_dt=date(2026, 4, 7),
        scraped_date="2026-04-07",
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    defaults.update(overrides)
    return Paper(**defaults)


class ScrapeApiQaTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    @patch("app.routes.api.SCRAPE_JOB_MANAGER.start_or_get_active")
    def test_trigger_scrape_forwards_force_flag(self, mock_start):
        mock_start.return_value = SimpleNamespace(
            id="job-force",
            status="running",
            started_at=datetime(2026, 4, 7, 9, 0, 0),
        )

        response = self.client.post(
            "/api/scrape",
            json={"force": True},
            headers={"X-CSRF-Token": self._csrf_token()},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["job_id"], "job-force")
        mock_start.assert_called_once_with(self.app, force=True)

    @patch("app.services.scrape_engine.execute_historical_scrape")
    def test_search_historical_forwards_categories_and_dates(self, mock_historical):
        mock_historical.return_value = {
            "new_papers": 2,
            "duplicates_skipped": 1,
            "total_matched": 3,
            "total_in_feed": 4,
        }

        response = self.client.post(
            "/api/search/historical",
            json={
                "categories": ["cs.CV", "cs.AI"],
                "start_date": "2026-04-01",
                "end_date": "2026-04-03",
            },
            headers={"X-CSRF-Token": self._csrf_token()},
        )

        self.assertEqual(response.status_code, 200)
        mock_historical.assert_called_once_with(
            self.app,
            ["cs.CV", "cs.AI"],
            date(2026, 4, 1),
            date(2026, 4, 3),
        )
        self.assertEqual(response.get_json()["new_papers"], 2)

    def test_search_historical_rejects_missing_dates(self):
        response = self.client.post(
            "/api/search/historical",
            json={"categories": ["cs.CV"]},
            headers={"X-CSRF-Token": self._csrf_token()},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("required", response.get_json()["error"])


class ScrapeRunQaTests(FlaskDBTestCase):
    def test_execute_scrape_records_forced_successful_run(self):
        from app.services.scrape_engine import execute_scrape

        with (
            patch("app.services.scrape_engine.parse_feed_entries", return_value=[]),
            patch("app.services.scrape_engine.enrich_entries_with_api_metadata"),
            patch("app.services.scrape_engine._process_entries_with_pipeline", return_value=iter([])),
        ):
            result = execute_scrape(self.app, force=True)

        self.assertEqual(result["new_papers"], 0)
        run = ScrapeRun.query.one()
        self.assertEqual(run.status, "success")
        self.assertTrue(run.forced)
        self.assertIsNotNone(run.started_at)
        self.assertIsNotNone(run.finished_at)

    def test_execute_scrape_records_error_run_on_failure(self):
        from app.services.scrape_engine import execute_scrape

        with patch("app.services.scrape_engine.parse_feed_entries", side_effect=RuntimeError("feed boom")):
            with self.assertRaises(RuntimeError):
                execute_scrape(self.app, force=True)

        run = ScrapeRun.query.one()
        self.assertEqual(run.status, "error")
        self.assertTrue(run.forced)
        self.assertIsNotNone(run.finished_at)


class ScrapeSseQaTests(FlaskDBTestCase):
    def test_stream_for_request_formats_sse_events(self):
        manager = ScrapeJobManager()

        def fake_scrape(app, event_callback=None, force=False):
            event_callback("status", {"phase": "processing", "message": "Working..."})
            event_callback("done", {"new_papers": 1, "duplicates_skipped": 0, "total_matched": 1, "total_in_feed": 2})
            return {}

        with patch("app.services.jobs.execute_scrape", side_effect=fake_scrape):
            stream = "".join(manager.stream_for_request(self.app, force=True))

        self.assertIn("event: status", stream)
        self.assertIn('"phase": "processing"', stream)
        self.assertIn("event: done", stream)
        self.assertIn('"new_papers": 1', stream)


class ScrapeCliQaTests(FlaskDBTestCase):
    @patch("app.cli.scrape.create_app")
    @patch("app.cli.scrape.run_scrape")
    def test_cli_scrape_prints_matched_paper_metadata_from_persisted_db(self, mock_run_scrape, mock_create_app):
        fake_app = self.app
        mock_create_app.return_value = fake_app

        def fake_run(app):
            with app.app_context():
                db.session.add(_make_paper())
                db.session.commit()
            return {
                "new_papers": 1,
                "duplicates_skipped": 0,
                "total_matched": 1,
                "total_in_feed": 2,
            }

        mock_run_scrape.side_effect = fake_run

        from app.cli.scrape import main

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            main()

        output = stdout.getvalue()
        self.assertIn("===== Matched Articles =====", output)
        self.assertIn("New: 1 | Duplicates skipped: 0 | Total matched: 1 / 2", output)
        self.assertIn("Title: CLI Matched Paper", output)
        self.assertIn("Authors: Alice Example, Bob Example", output)
        self.assertIn("ArXiv Link: https://arxiv.org/abs/2604.00001", output)
        self.assertIn("PDF Link: https://arxiv.org/pdf/2604.00001", output)
        self.assertIn("Paper Score: 17.50", output)
        self.assertIn("Matched Terms:", output)
        self.assertEqual(Paper.query.count(), 1)


if __name__ == "__main__":
    unittest.main()
