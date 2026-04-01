"""Tests for Zotero-inspired improvement features."""

from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from app.models import Collection, FeedSource, Paper, PaperCollection, SavedSearch, db
from tests.helpers import FlaskDBTestCase


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        arxiv_id="2603.12345",
        title="Test Paper on Vision",
        authors="Alice Smith, Bob Jones",
        link="https://arxiv.org/abs/2603.12345",
        pdf_link="https://arxiv.org/pdf/2603.12345",
        abstract_text="An abstract about vision transformers.",
        summary_text="A summary.",
        topic_tags=["vision"],
        categories=["cs.CV"],
        resource_links=[],
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=10.0,
        feedback_score=0,
        is_hidden=False,
        publication_date="2026-03-13",
        scraped_date="2026-03-13",
        publication_dt=date(2026, 3, 13),
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    defaults.update(overrides)
    return Paper(**defaults)


class _CsrfMixin:
    """Mixin to get a valid CSRF token for API tests."""

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]


class ReadingStatusAPITests(FlaskDBTestCase, _CsrfMixin):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_set_reading_status(self):
        paper = _make_paper()
        db.session.add(paper)
        db.session.commit()

        response = self.client.post(
            f"/api/papers/{paper.id}/reading-status",
            json={"status": "to_read"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["reading_status"], "to_read")

        refreshed = db.session.get(Paper, paper.id)
        self.assertEqual(refreshed.reading_status, "to_read")

    def test_clear_reading_status(self):
        paper = _make_paper(reading_status="reading")
        db.session.add(paper)
        db.session.commit()

        response = self.client.post(
            f"/api/papers/{paper.id}/reading-status",
            json={"status": None},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.get_json()["reading_status"])

    def test_invalid_reading_status(self):
        paper = _make_paper()
        db.session.add(paper)
        db.session.commit()

        response = self.client.post(
            f"/api/papers/{paper.id}/reading-status",
            json={"status": "invalid"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 400)


class UserNotesAPITests(FlaskDBTestCase, _CsrfMixin):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_save_notes(self):
        paper = _make_paper()
        db.session.add(paper)
        db.session.commit()

        response = self.client.put(
            f"/api/papers/{paper.id}/notes",
            json={"notes": "Important paper for thesis"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["user_notes"], "Important paper for thesis")

    def test_notes_404(self):
        response = self.client.put(
            "/api/papers/99999/notes",
            json={"notes": "test"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 404)


class UserTagsAPITests(FlaskDBTestCase, _CsrfMixin):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_add_tag(self):
        paper = _make_paper()
        db.session.add(paper)
        db.session.commit()

        response = self.client.post(
            f"/api/papers/{paper.id}/tags",
            json={"tag": "thesis-ch3"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("thesis-ch3", response.get_json()["user_tags"])

    def test_add_duplicate_tag(self):
        paper = _make_paper(user_tags=["existing"])
        db.session.add(paper)
        db.session.commit()

        response = self.client.post(
            f"/api/papers/{paper.id}/tags",
            json={"tag": "existing"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["user_tags"].count("existing"), 1)

    def test_remove_tag(self):
        paper = _make_paper(user_tags=["tag1", "tag2"])
        db.session.add(paper)
        db.session.commit()

        response = self.client.delete(
            f"/api/papers/{paper.id}/tags",
            json={"tag": "tag1"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("tag1", response.get_json()["user_tags"])
        self.assertIn("tag2", response.get_json()["user_tags"])

    def test_empty_tag_rejected(self):
        paper = _make_paper()
        db.session.add(paper)
        db.session.commit()

        response = self.client.post(
            f"/api/papers/{paper.id}/tags",
            json={"tag": ""},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 400)


class CollectionAPITests(FlaskDBTestCase, _CsrfMixin):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_create_collection(self):
        response = self.client.post(
            "/api/collections",
            json={"name": "Thesis Papers"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["name"], "Thesis Papers")

    def test_list_collections(self):
        db.session.add(Collection(name="A"))
        db.session.add(Collection(name="B"))
        db.session.commit()

        response = self.client.get("/api/collections")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()), 2)

    def test_add_paper_to_collection(self):
        c = Collection(name="Test")
        paper = _make_paper()
        db.session.add_all([c, paper])
        db.session.commit()

        response = self.client.post(
            f"/api/collections/{c.id}/papers",
            json={"paper_id": paper.id},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["added"], 1)

        # Verify it's in the collection.
        pc = PaperCollection.query.filter_by(paper_id=paper.id, collection_id=c.id).first()
        self.assertIsNotNone(pc)

    def test_remove_paper_from_collection(self):
        c = Collection(name="Test")
        paper = _make_paper()
        db.session.add_all([c, paper])
        db.session.commit()
        db.session.add(PaperCollection(paper_id=paper.id, collection_id=c.id))
        db.session.commit()

        response = self.client.delete(
            f"/api/collections/{c.id}/papers/{paper.id}",
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)

    def test_delete_collection(self):
        c = Collection(name="ToDelete")
        db.session.add(c)
        db.session.commit()

        response = self.client.delete(
            f"/api/collections/{c.id}",
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(db.session.get(Collection, c.id))

    def test_duplicate_collection_name(self):
        db.session.add(Collection(name="Existing"))
        db.session.commit()

        response = self.client.post(
            "/api/collections",
            json={"name": "Existing"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 409)


class SavedSearchAPITests(FlaskDBTestCase, _CsrfMixin):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_create_saved_search(self):
        response = self.client.post(
            "/api/saved-searches",
            json={"name": "Vision papers", "filters": {"q": "vision", "timeframe": "weekly"}},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["name"], "Vision papers")

    def test_list_saved_searches(self):
        db.session.add(SavedSearch(name="S1", filters={"q": "test"}))
        db.session.commit()

        response = self.client.get("/api/saved-searches")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()), 1)

    def test_delete_saved_search(self):
        s = SavedSearch(name="ToDelete", filters={})
        db.session.add(s)
        db.session.commit()

        response = self.client.delete(
            f"/api/saved-searches/{s.id}",
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)


class BulkOperationsAPITests(FlaskDBTestCase, _CsrfMixin):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_bulk_feedback(self):
        p1 = _make_paper(arxiv_id="2603.00001", link="https://arxiv.org/abs/2603.00001")
        p2 = _make_paper(arxiv_id="2603.00002", link="https://arxiv.org/abs/2603.00002")
        db.session.add_all([p1, p2])
        db.session.commit()

        response = self.client.post(
            "/api/papers/bulk-feedback",
            json={"paper_ids": [p1.id, p2.id], "action": "save"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["processed"], 2)

    def test_bulk_bibtex(self):
        p1 = _make_paper(arxiv_id="2603.00001", link="https://arxiv.org/abs/2603.00001")
        db.session.add(p1)
        db.session.commit()

        response = self.client.get(f"/api/papers/bulk-bibtex?ids={p1.id}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("@article{", response.get_data(as_text=True))


class AuthorSearchAPITests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_search_authors(self):
        p1 = _make_paper(authors="Alice Smith, Bob Jones")
        db.session.add(p1)
        db.session.commit()

        response = self.client.get("/api/authors?q=Alice")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(any("Alice Smith" in a["name"] for a in data))

    def test_empty_author_search(self):
        response = self.client.get("/api/authors?q=")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), [])


class PaperGraphAPITests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_paper_graph(self):
        p1 = _make_paper(title="Vision Transformers for Object Detection")
        db.session.add(p1)
        db.session.commit()

        response = self.client.get(f"/api/papers/{p1.id}/graph")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)

    def test_paper_graph_404(self):
        response = self.client.get("/api/papers/99999/graph")
        self.assertEqual(response.status_code, 404)


class DuplicateDetectionTests(unittest.TestCase):
    def test_find_duplicates_exact(self):
        from app.services.related import find_duplicates

        titles = {1: "Vision Transformers for Object Detection"}
        dups = find_duplicates("Vision Transformers for Object Detection", titles)
        self.assertTrue(len(dups) > 0)
        self.assertEqual(dups[0][0], 1)
        self.assertAlmostEqual(dups[0][1], 1.0, places=2)

    def test_find_duplicates_no_match(self):
        from app.services.related import find_duplicates

        titles = {1: "Quantum Computing Fundamentals"}
        dups = find_duplicates("Vision Transformers for Object Detection", titles)
        self.assertEqual(len(dups), 0)


class FeedSourceAPITests(FlaskDBTestCase, _CsrfMixin):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_create_feed_source(self):
        response = self.client.post(
            "/api/feed-sources",
            json={"name": "CS.AI", "url": "https://rss.arxiv.org/rss/cs.AI"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 201)

    def test_list_feed_sources(self):
        db.session.add(FeedSource(name="Test", url="https://example.com/rss"))
        db.session.commit()

        response = self.client.get("/api/feed-sources")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()), 1)

    def test_toggle_feed_source(self):
        s = FeedSource(name="Test", url="https://example.com/rss")
        db.session.add(s)
        db.session.commit()

        response = self.client.post(
            f"/api/feed-sources/{s.id}/toggle",
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()["enabled"])

    def test_delete_feed_source(self):
        s = FeedSource(name="Test", url="https://example.com/rss")
        db.session.add(s)
        db.session.commit()

        response = self.client.delete(
            f"/api/feed-sources/{s.id}",
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 200)


class DashboardFilterTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_reading_status_filter(self):
        p1 = _make_paper(arxiv_id="2603.00001", link="https://arxiv.org/abs/2603.00001", reading_status="to_read")
        p2 = _make_paper(arxiv_id="2603.00002", link="https://arxiv.org/abs/2603.00002", reading_status=None)
        db.session.add_all([p1, p2])
        db.session.commit()

        response = self.client.get("/?reading_status=to_read&timeframe=all")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"2603.00001", response.data)
        self.assertNotIn(b"2603.00002", response.data)

    def test_author_filter(self):
        p1 = _make_paper(arxiv_id="2603.00001", link="https://arxiv.org/abs/2603.00001", authors="Alice Smith")
        p2 = _make_paper(arxiv_id="2603.00002", link="https://arxiv.org/abs/2603.00002", authors="Bob Jones")
        db.session.add_all([p1, p2])
        db.session.commit()

        response = self.client.get("/?author=Alice+Smith&timeframe=all")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Alice Smith", response.data)

    def test_collection_filter(self):
        c = Collection(name="TestCol")
        paper = _make_paper()
        db.session.add_all([c, paper])
        db.session.commit()
        db.session.add(PaperCollection(paper_id=paper.id, collection_id=c.id))
        db.session.commit()

        response = self.client.get(f"/?collection={c.id}&timeframe=all")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Test Paper on Vision", response.data)

    def test_recommended_sort(self):
        p1 = _make_paper(arxiv_id="2603.00001", link="https://arxiv.org/abs/2603.00001", recommendation_score=9.0)
        p2 = _make_paper(arxiv_id="2603.00002", link="https://arxiv.org/abs/2603.00002", recommendation_score=3.0)
        db.session.add_all([p1, p2])
        db.session.commit()

        response = self.client.get("/?sort=recommended&timeframe=all")
        self.assertEqual(response.status_code, 200)

    def test_abstract_search(self):
        p1 = _make_paper(
            arxiv_id="2603.00001",
            link="https://arxiv.org/abs/2603.00001",
            abstract_text="Novel approach using transformers for segmentation",
        )
        db.session.add(p1)
        db.session.commit()

        response = self.client.get("/?q=segmentation&timeframe=all")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"2603.00001", response.data)


class SchedulerTests(unittest.TestCase):
    def test_scheduler_next_run(self):
        from app.services.scheduler import ScrapeScheduler

        scheduler = ScrapeScheduler()
        self.assertFalse(scheduler.is_enabled)
        self.assertIsNone(scheduler.next_run_at)

    def test_scheduler_schedule_time(self):
        from app.services.scheduler import ScrapeScheduler

        scheduler = ScrapeScheduler()
        self.assertEqual(scheduler.schedule_time, "08:00")
