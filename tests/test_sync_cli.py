from __future__ import annotations

import unittest
from datetime import date, datetime
from unittest.mock import patch

from app.models import SyncState
from sync_cli import chunk_end_timestamp, iter_date_chunks, run_sync, upsert_sync_state
from tests.helpers import FlaskDBTestCase


class SyncCliChunkingTests(unittest.TestCase):
    def test_iter_date_chunks_splits_range_into_weeks(self):
        chunks = list(iter_date_chunks(date(2026, 1, 1), date(2026, 1, 17)))

        self.assertEqual(
            chunks,
            [
                (date(2026, 1, 1), date(2026, 1, 7)),
                (date(2026, 1, 8), date(2026, 1, 14)),
                (date(2026, 1, 15), date(2026, 1, 17)),
            ],
        )

    def test_chunk_end_timestamp_uses_end_of_day(self):
        self.assertEqual(
            chunk_end_timestamp(date(2026, 1, 7)),
            datetime(2026, 1, 7, 23, 59, 59, 999999),
        )


class SyncCliStateTests(FlaskDBTestCase):
    @patch("sync_cli.now_utc", return_value=datetime(2026, 1, 8, 9, 30, 0))
    def test_upsert_sync_state_creates_and_updates_progress(self, _mock_now):
        upsert_sync_state("cs.CV", synced_through=date(2026, 1, 7), paper_count=12)

        stored = SyncState.query.filter_by(category="cs.CV").one()
        self.assertEqual(stored.last_synced_submitted_at, datetime(2026, 1, 7, 23, 59, 59, 999999))
        self.assertEqual(stored.last_synced_updated_at, datetime(2026, 1, 8, 9, 30, 0))
        self.assertEqual(stored.last_synced_paper_count, 12)
        self.assertIsNone(stored.last_cursor_page)
        self.assertIsNone(stored.last_cursor_arxiv_id)

    @patch("sync_cli.execute_historical_scrape")
    def test_run_sync_executes_historical_scrape_inside_app_context(self, mock_historical_scrape):
        def _fake_execute(app, categories, start_dt, end_dt):
            from flask import has_app_context

            self.assertTrue(has_app_context())
            return {
                "new_papers": 0,
                "duplicates_skipped": 0,
                "total_matched": 0,
                "total_in_feed": 0,
            }

        mock_historical_scrape.side_effect = _fake_execute

        run_sync(
            self.app,
            category="cs.CV",
            start_dt=date(2026, 1, 1),
            end_dt=date(2026, 1, 1),
            emit=lambda _message: None,
        )

        self.assertEqual(mock_historical_scrape.call_count, 1)

    @patch("sync_cli.now_utc", return_value=datetime(2026, 1, 15, 10, 0, 0))
    @patch("sync_cli.execute_historical_scrape")
    def test_run_sync_processes_chunks_and_updates_state(self, mock_historical_scrape, _mock_now):
        mock_historical_scrape.side_effect = [
            {
                "new_papers": 2,
                "duplicates_skipped": 1,
                "total_matched": 3,
                "total_in_feed": 5,
            },
            {
                "new_papers": 1,
                "duplicates_skipped": 0,
                "total_matched": 2,
                "total_in_feed": 4,
            },
        ]
        messages: list[str] = []

        summary = run_sync(
            self.app,
            category="cs.CV",
            start_dt=date(2026, 1, 1),
            end_dt=date(2026, 1, 10),
            chunk_days=7,
            emit=messages.append,
        )

        self.assertEqual(
            summary,
            {
                "new_papers": 3,
                "duplicates_skipped": 1,
                "total_matched": 5,
                "total_in_feed": 9,
            },
        )
        self.assertEqual(mock_historical_scrape.call_count, 2)
        self.assertEqual(
            mock_historical_scrape.call_args_list[0].args[1:],
            (["cs.CV"], date(2026, 1, 1), date(2026, 1, 7)),
        )
        self.assertEqual(
            mock_historical_scrape.call_args_list[1].args[1:],
            (["cs.CV"], date(2026, 1, 8), date(2026, 1, 10)),
        )

        stored = SyncState.query.filter_by(category="cs.CV").one()
        self.assertEqual(stored.last_synced_submitted_at, datetime(2026, 1, 10, 23, 59, 59, 999999))
        self.assertEqual(stored.last_synced_updated_at, datetime(2026, 1, 15, 10, 0, 0))
        self.assertEqual(stored.last_synced_paper_count, 4)
        self.assertIsNone(stored.last_cursor_page)
        self.assertIsNone(stored.last_cursor_arxiv_id)
        self.assertTrue(messages[0].startswith("Starting sync for cs.CV"))
        self.assertTrue(messages[-1].startswith("Sync complete:"))


if __name__ == "__main__":
    unittest.main()
