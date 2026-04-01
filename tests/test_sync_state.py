from __future__ import annotations

import unittest
from datetime import datetime

from sqlalchemy import inspect, text

from app.models import SyncState, db
from app.schema import ensure_schema
from tests.helpers import FlaskDBTestCase


class SyncStateModelTests(FlaskDBTestCase):
    def test_sync_state_persists_incremental_progress(self):
        state = SyncState(
            category="cs.CV",
            last_synced_submitted_at=datetime(2026, 1, 1, 8, 0, 0),
            last_synced_updated_at=datetime(2026, 1, 2, 9, 30, 0),
            last_synced_paper_count=12,
        )

        db.session.add(state)
        db.session.commit()

        stored = SyncState.query.filter_by(category="cs.CV").one()
        self.assertEqual(stored.last_synced_paper_count, 12)
        self.assertEqual(stored.last_synced_submitted_at, datetime(2026, 1, 1, 8, 0, 0))
        self.assertEqual(stored.last_synced_updated_at, datetime(2026, 1, 2, 9, 30, 0))
        self.assertIsNotNone(stored.updated_at)

    def test_sync_state_defaults_count_to_zero(self):
        state = SyncState(category="cs.LG")

        db.session.add(state)
        db.session.commit()

        stored = SyncState.query.filter_by(category="cs.LG").one()
        self.assertEqual(stored.last_synced_paper_count, 0)


class SyncStateSchemaTests(FlaskDBTestCase):
    def test_ensure_schema_recreates_missing_sync_state_table(self):
        db.session.execute(text("DROP TABLE sync_state"))
        db.session.commit()

        self.assertNotIn("sync_state", inspect(db.engine).get_table_names())

        ensure_schema()

        self.assertIn("sync_state", inspect(db.engine).get_table_names())


if __name__ == "__main__":
    unittest.main()
