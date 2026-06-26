"""QA round 5 regression tests for CLI hardening.

- R5-con8 (S2): a non-positive --batch-size makes the offset-paginated backfill
  loops spin forever (SQLite LIMIT -1 = unlimited, offset never advances). The
  argparse type must reject it.
- R5-con9 (S3): an oversized --chunk-days overflows date/timedelta arithmetic in
  iter_date_chunks (OverflowError, not ValueError); the sync CLI must report it
  cleanly instead of dumping a traceback.
"""

from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

from app.cli.backfill import build_parser
from app.cli.sync import iter_date_chunks
from app.cli.sync import main as sync_main


class BatchSizeValidationTests(unittest.TestCase):
    def test_negative_batch_size_rejected(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["embeddings", "--batch-size", "-1"])

    def test_zero_batch_size_rejected(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["index-rebuild", "--batch-size", "0"])

    def test_positive_batch_size_accepted(self):
        parser = build_parser()
        args = parser.parse_args(["embeddings", "--batch-size", "32"])
        self.assertEqual(args.batch_size, 32)


class ChunkDaysOverflowTests(unittest.TestCase):
    def test_iter_date_chunks_overflows_on_huge_chunk_days(self):
        with self.assertRaises(OverflowError):
            list(iter_date_chunks(date(2020, 1, 1), date(2020, 2, 1), chunk_days=10**18))

    @patch("app.cli.sync.run_sync", side_effect=OverflowError("date value out of range"))
    @patch("app.cli.sync.create_app", return_value=object())
    def test_main_reports_overflow_cleanly(self, _create_app, _run_sync):
        rc = sync_main(
            ["--from", "2020-01-01", "--to", "2020-02-01", "--category", "cs.CV", "--chunk-days", "999999999999999"]
        )
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
