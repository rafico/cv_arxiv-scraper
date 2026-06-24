"""QA round 4 regression tests for Collections API.

Covers G4 (S2): update_collection() renaming a collection to another
collection's name must return 409, not an unhandled 500 (Collection.name is
UNIQUE). Also verifies renaming to the same name is a no-op (no false 409).
"""

from __future__ import annotations

from app.models import Collection, db
from tests.helpers import FlaskDBTestCase


class UpdateCollectionDuplicateNameTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_g4_rename_to_existing_name_returns_409_not_500(self):
        a = Collection(name="Alpha")
        b = Collection(name="Beta")
        db.session.add_all([a, b])
        db.session.commit()
        b_id = b.id

        response = self.client.put(
            f"/api/collections/{b_id}",
            json={"name": "Alpha"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 409)
        # Session must not be left in a failed state; the original name stands.
        refreshed = db.session.get(Collection, b_id)
        self.assertEqual(refreshed.name, "Beta")

    def test_g4_rename_to_same_name_is_noop_not_409(self):
        c = Collection(name="Gamma")
        db.session.add(c)
        db.session.commit()
        c_id = c.id

        response = self.client.put(
            f"/api/collections/{c_id}",
            json={"name": "Gamma", "description": "updated"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        refreshed = db.session.get(Collection, c_id)
        self.assertEqual(refreshed.name, "Gamma")
        self.assertEqual(refreshed.description, "updated")


if __name__ == "__main__":
    import unittest

    unittest.main()
