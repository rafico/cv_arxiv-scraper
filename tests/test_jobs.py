"""Tests for ScrapeJobManager error handling and event contracts."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tests.helpers import FlaskDBTestCase


class ScrapeJobErrorTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        # Use a fresh manager per test to avoid cross-test state.
        from app.services.jobs import ScrapeJobManager

        self.manager = ScrapeJobManager()

    def test_error_sets_status_to_error_not_finished(self):
        """When execute_scrape raises, job.status must be 'error', not 'finished'."""
        with patch(
            "app.services.jobs.execute_scrape",
            side_effect=RuntimeError("boom"),
        ):
            job = self.manager.start_or_get_active(self.app)
            events = list(self.manager.stream_events(job.id, heartbeat_seconds=1))

        self.assertEqual(job.status, "error")
        self.assertIsNotNone(job.finished_at)

        event_types = [e for e, _ in events]
        self.assertIn("scrape_error", event_types)
        self.assertNotIn("done", event_types)

    def test_success_sets_status_to_finished(self):
        """Normal completion should set status to 'finished' with a 'done' event."""

        def fake_scrape(app, event_callback=None):
            event_callback("done", {"new_papers": 1, "duplicates_skipped": 0, "total_matched": 1, "total_in_feed": 5})
            return {}

        with patch("app.services.jobs.execute_scrape", side_effect=fake_scrape):
            job = self.manager.start_or_get_active(self.app)
            events = list(self.manager.stream_events(job.id, heartbeat_seconds=1))

        self.assertEqual(job.status, "finished")
        event_types = [e for e, _ in events]
        self.assertIn("done", event_types)
        self.assertNotIn("scrape_error", event_types)

    def test_stream_events_missing_job_yields_scrape_error_only(self):
        """Streaming a non-existent job should yield scrape_error and stop."""
        events = list(self.manager.stream_events("nonexistent"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], "scrape_error")


class ScrapeStatusEndpointTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_status_returns_not_running_when_idle(self):
        response = self.client.get("/api/scrape/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertFalse(data["running"])


if __name__ == "__main__":
    unittest.main()
