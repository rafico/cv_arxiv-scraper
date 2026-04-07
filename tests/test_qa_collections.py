"""QA tests for Collections API.

Covers: CVARX-60 (Collections)
- Create collection with name, description, color
- List collections with paper counts
- Update collection
- Delete collection (papers remain)
- Add/remove papers from collection
- Empty collection behavior
- Duplicate collection name rejection
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from app.models import Collection, Paper, PaperCollection, db
from tests.helpers import FlaskDBTestCase


def _make_paper(idx: int = 0, **overrides) -> Paper:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    defaults = dict(
        arxiv_id=f"2607.{2000 + idx:04d}",
        title=f"Collection Test Paper {idx}",
        authors="Author A",
        link=f"https://arxiv.org/abs/2607.{2000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2607.{2000 + idx:04d}",
        abstract_text="abstract",
        summary_text="summary",
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=10.0,
        feedback_score=0,
        is_hidden=False,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=now,
    )
    defaults.update(overrides)
    return Paper(**defaults)


class CollectionsCRUDTests(FlaskDBTestCase):
    """Test collection CRUD operations."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_create_collection_with_name_only(self):
        response = self.client.post(
            "/api/collections",
            json={"name": "My Papers"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertEqual(data["name"], "My Papers")

    def test_create_collection_with_description_and_color(self):
        response = self.client.post(
            "/api/collections",
            json={"name": "Thesis", "description": "Papers for thesis", "color": "#FF5733"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertEqual(data["name"], "Thesis")

    def test_list_collections_with_paper_counts(self):
        c = Collection(name="Test Collection")
        p = _make_paper(0)
        db.session.add_all([c, p])
        db.session.commit()
        db.session.add(PaperCollection(paper_id=p.id, collection_id=c.id))
        db.session.commit()

        response = self.client.get("/api/collections")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["name"], "Test Collection")
        self.assertEqual(data[0]["paper_count"], 1)

    def test_update_collection(self):
        c = Collection(name="Old Name")
        db.session.add(c)
        db.session.commit()

        response = self.client.put(
            f"/api/collections/{c.id}",
            json={"name": "New Name", "description": "Updated desc"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        refreshed = db.session.get(Collection, c.id)
        self.assertEqual(refreshed.name, "New Name")

    def test_delete_collection_papers_remain(self):
        c = Collection(name="To Delete")
        p = _make_paper(0)
        db.session.add_all([c, p])
        db.session.commit()
        db.session.add(PaperCollection(paper_id=p.id, collection_id=c.id))
        db.session.commit()
        paper_id = p.id

        response = self.client.delete(
            f"/api/collections/{c.id}",
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        # Paper still exists
        self.assertIsNotNone(db.session.get(Paper, paper_id))

    def test_duplicate_collection_name_rejected(self):
        db.session.add(Collection(name="Existing"))
        db.session.commit()

        response = self.client.post(
            "/api/collections",
            json={"name": "Existing"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 409)

    def test_add_paper_to_collection(self):
        c = Collection(name="Test")
        p = _make_paper(0)
        db.session.add_all([c, p])
        db.session.commit()

        response = self.client.post(
            f"/api/collections/{c.id}/papers",
            json={"paper_id": p.id},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        pc = PaperCollection.query.filter_by(paper_id=p.id, collection_id=c.id).first()
        self.assertIsNotNone(pc)

    def test_add_paper_to_nonexistent_collection(self):
        p = _make_paper(0)
        db.session.add(p)
        db.session.commit()

        response = self.client.post(
            "/api/collections/9999/papers",
            json={"paper_id": p.id},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 404)

    def test_remove_paper_from_collection(self):
        c = Collection(name="Test")
        p = _make_paper(0)
        db.session.add_all([c, p])
        db.session.commit()
        db.session.add(PaperCollection(paper_id=p.id, collection_id=c.id))
        db.session.commit()

        response = self.client.delete(
            f"/api/collections/{c.id}/papers/{p.id}",
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        pc = PaperCollection.query.filter_by(paper_id=p.id, collection_id=c.id).first()
        self.assertIsNone(pc)

    def test_list_empty_collections(self):
        response = self.client.get("/api/collections")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), [])

    def test_collection_tab_shows_papers(self):
        c = Collection(name="MyCol")
        p = _make_paper(0, title="Special Paper In Collection")
        db.session.add_all([c, p])
        db.session.commit()
        db.session.add(PaperCollection(paper_id=p.id, collection_id=c.id))
        db.session.commit()

        response = self.client.get(f"/?collection={c.id}&timeframe=all")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Special Paper In Collection", response.data)


if __name__ == "__main__":
    import unittest

    unittest.main()
