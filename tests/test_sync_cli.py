from __future__ import annotations

import contextlib
import io
import unittest
from datetime import date, datetime
from unittest.mock import patch

from app.models import Collection, Paper, PaperCollection, PaperFeedback, SyncState, db
from app.services.ingest import PaperCandidate
from sync_cli import chunk_end_timestamp, iter_date_chunks, main, run_query_import, run_sync, upsert_sync_state
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


class SyncCliQueryTests(FlaskDBTestCase):
    @patch("app.services.embed_backfill.backfill_embeddings", return_value=0)
    @patch("sync_cli.upsert_sync_state")
    @patch("sync_cli.execute_historical_scrape")
    @patch("sync_cli.ArxivApiBackend.fetch")
    def test_query_imports_into_collection_and_leaves_sync_state_alone(self, fetch, scrape, upsert, _embed):
        db.session.add(
            Paper(
                arxiv_id="2601.00001",
                title="Already Stored",
                authors="Author A",
                link="https://arxiv.org/abs/2601.00001",
                pdf_link="https://arxiv.org/pdf/2601.00001",
                match_type="title",
                scraped_date="2026-01-01",
            )
        )
        db.session.commit()
        fetch.return_value = [
            PaperCandidate(arxiv_id="2601.00001", link="http://arxiv.org/abs/2601.00001v1", title="Already Stored"),
            PaperCandidate(
                arxiv_id="1905.00001",
                link="http://arxiv.org/abs/1905.00001v2",
                title="A Seminal Paper",
                authors_list=["Dana Seed"],
                abstract="Seed abstract.",
                publication_date="2019-05-01",
                categories=["cs.CV"],
            ),
        ]
        messages: list[str] = []

        stats = run_query_import(
            self.app,
            query='abs:"ovs"',
            collection="OVS",
            start_dt=date(2019, 1, 1),
            end_dt=date(2026, 9, 28),
            max_results=2,
            emit=messages.append,
        )

        self.assertEqual((stats["created"], stats["linked"]), (1, 1))
        self.assertEqual(fetch.call_args.kwargs["query"], 'abs:"ovs"')
        collection = Collection.query.filter_by(name="OVS").one()
        members = Paper.query.join(PaperCollection).filter(PaperCollection.collection_id == collection.id).all()
        self.assertEqual({p.arxiv_id for p in members}, {"2601.00001", "1905.00001"})
        seed = Paper.query.filter_by(arxiv_id="1905.00001").one()
        self.assertEqual(
            (seed.match_type, seed.authors, seed.publication_dt), ("import", "Dana Seed", date(2019, 5, 1))
        )
        self.assertEqual(SyncState.query.count(), 0)
        self.assertEqual(PaperFeedback.query.count(), 0)
        scrape.assert_not_called()
        upsert.assert_not_called()
        self.assertTrue(any(m.startswith("WARNING: hit --max-results 2") for m in messages))

    @patch("sync_cli.run_query_import")
    @patch("sync_cli.create_app", return_value=object())
    def test_main_validates_modes_and_defaults_query_window(self, _create_app, run_query):
        for argv in (
            ["--query", "ti:x"],  # no --collection
            ["--query", "ti:x", "--collection", "OVS", "--max-results", "0"],
            ["--category", "cs.CV"],  # sync mode still needs --from/--to
        ):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                main(argv)
        run_query.assert_not_called()

        self.assertEqual(main(["--query", "ti:x", "--collection", "OVS"]), 0)
        kwargs = run_query.call_args.kwargs
        self.assertEqual((kwargs["start_dt"], kwargs["categories"]), (date(1991, 1, 1), None))


if __name__ == "__main__":
    unittest.main()
