from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from app.models import Paper, SavedSearch, db
from app.services.search import search_semantic
from tests.helpers import FlaskDBTestCase


def _make_paper(idx: int = 0, **overrides) -> Paper:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    defaults = dict(
        arxiv_id=f"2607.{1000 + idx:04d}",
        title=f"Search QA Paper {idx}",
        authors="Author A, Author B",
        link=f"https://arxiv.org/abs/2607.{1000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2607.{1000 + idx:04d}",
        abstract_text="abstract text about vision transformers",
        summary_text="Summary text",
        topic_tags=["Segmentation"],
        categories=["cs.CV"],
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=10.0 + idx,
        feedback_score=0,
        is_hidden=False,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=now,
    )
    defaults.update(overrides)
    return Paper(**defaults)


class SearchApiQaTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_keyword_search_uses_default_limit_and_preserves_bm25_order(self):
        first = _make_paper(1, title="First BM25 Match")
        second = _make_paper(2, title="Best BM25 Match")
        db.session.add_all([first, second])
        db.session.commit()

        with patch("app.services.search.search_bm25", return_value=[(second.id, 9.5), (first.id, 4.0)]) as mock_bm25:
            response = self.client.get("/api/search?q=vision&mode=keyword")

        self.assertEqual(response.status_code, 200)
        mock_bm25.assert_called_once_with("vision", limit=30)
        data = response.get_json()
        self.assertEqual([result["title"] for result in data["results"]], ["Best BM25 Match", "First BM25 Match"])
        self.assertEqual([result["score"] for result in data["results"]], [9.5, 4.0])

    def test_keyword_search_caps_limit_at_100(self):
        with patch("app.services.search.search_bm25", return_value=[]) as mock_bm25:
            response = self.client.get("/api/search?q=vision&mode=keyword&limit=999")

        self.assertEqual(response.status_code, 200)
        mock_bm25.assert_called_once_with("vision", limit=100)

    def test_semantic_search_returns_empty_results_when_index_missing(self):
        fake_service = SimpleNamespace(index_count=lambda: 0)

        with patch("app.services.embeddings.get_embedding_service", return_value=fake_service):
            self.assertEqual(search_semantic("vision transformer"), [])

        response = self.client.get("/api/search?q=vision&mode=semantic")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["results"], [])

    def test_hybrid_search_enriches_rrf_results_with_paper_data(self):
        paper = _make_paper(3, title="Hybrid Search Match", authors="Jane Doe")
        db.session.add(paper)
        db.session.commit()

        with patch(
            "app.services.search.search_hybrid",
            return_value=[{"paper_id": paper.id, "rrf_score": 0.42, "bm25_rank": 2, "semantic_rank": 1}],
        ) as mock_hybrid:
            response = self.client.get("/api/search?q=hybrid&mode=hybrid&limit=5")

        self.assertEqual(response.status_code, 200)
        mock_hybrid.assert_called_once_with("hybrid", top_k=5)
        result = response.get_json()["results"][0]
        self.assertEqual(result["title"], "Hybrid Search Match")
        self.assertEqual(result["authors"], "Jane Doe")
        self.assertEqual(result["rrf_score"], 0.42)
        self.assertEqual(result["semantic_rank"], 1)


class SavedSearchApiQaTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        self.client.get("/")
        with self.client.session_transaction() as session:
            self.csrf_token = session["settings_csrf_token"]

    def test_saved_search_flags_round_trip_through_create_update_and_list(self):
        response = self.client.post(
            "/api/saved-searches",
            json={
                "name": "Tracked Search",
                "categories": ["cs.CV"],
                "is_active": False,
                "notify_on_match": True,
                "include_keywords": ["transformer"],
            },
            headers={"X-CSRF-Token": self.csrf_token},
        )

        self.assertEqual(response.status_code, 201)
        created = response.get_json()
        self.assertFalse(created["is_active"])
        self.assertTrue(created["notify_on_match"])
        self.assertEqual(created["categories"], ["cs.CV"])

        update_response = self.client.put(
            f"/api/saved-searches/{created['id']}",
            json={"is_active": True, "notify_on_match": False},
            headers={"X-CSRF-Token": self.csrf_token},
        )

        self.assertEqual(update_response.status_code, 200)
        updated = update_response.get_json()
        self.assertTrue(updated["is_active"])
        self.assertFalse(updated["notify_on_match"])

        list_response = self.client.get("/api/saved-searches")
        self.assertEqual(list_response.status_code, 200)
        listed = {item["id"]: item for item in list_response.get_json()}
        self.assertTrue(listed[created["id"]]["is_active"])
        self.assertFalse(listed[created["id"]]["notify_on_match"])

        delete_response = self.client.delete(
            f"/api/saved-searches/{created['id']}",
            headers={"X-CSRF-Token": self.csrf_token},
        )
        self.assertEqual(delete_response.status_code, 200)
        self.assertTrue(delete_response.get_json()["deleted"])
        self.assertIsNone(db.session.get(SavedSearch, created["id"]))

    def test_saved_search_run_honors_category_and_filter_bundle(self):
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        matching = _make_paper(
            10,
            title="Transformer Methods for Vision",
            authors="Jane Doe, Alex Smith",
            categories=["cs.CV"],
            citation_count=25,
            publication_dt=date.today(),
            abstract_text="Transformer methods for segmentation",
            scraped_at=now,
        )
        wrong_category = _make_paper(
            11,
            title="Transformer Methods for Vision",
            authors="Jane Doe",
            categories=["cs.AI"],
            citation_count=25,
            publication_dt=date.today(),
            abstract_text="Transformer methods for segmentation",
            scraped_at=now,
        )
        excluded_keyword = _make_paper(
            12,
            title="Transformer Survey for Vision",
            authors="Jane Doe",
            categories=["cs.CV"],
            citation_count=25,
            publication_dt=date.today(),
            abstract_text="A survey of transformer methods",
            scraped_at=now,
        )
        old_paper = _make_paper(
            13,
            title="Transformer Methods for Vision",
            authors="Jane Doe",
            categories=["cs.CV"],
            citation_count=25,
            publication_dt=date.today() - timedelta(days=90),
            abstract_text="Transformer methods for segmentation",
            scraped_at=now - timedelta(days=90),
        )
        low_citations = _make_paper(
            14,
            title="Transformer Methods for Vision",
            authors="Jane Doe",
            categories=["cs.CV"],
            citation_count=3,
            publication_dt=date.today(),
            abstract_text="Transformer methods for segmentation",
            scraped_at=now,
        )
        wrong_author = _make_paper(
            15,
            title="Transformer Methods for Vision",
            authors="Someone Else",
            categories=["cs.CV"],
            citation_count=25,
            publication_dt=date.today(),
            abstract_text="Transformer methods for segmentation",
            scraped_at=now,
        )
        db.session.add_all([matching, wrong_category, excluded_keyword, old_paper, low_citations, wrong_author])
        db.session.commit()

        create_response = self.client.post(
            "/api/saved-searches",
            json={
                "name": "Full Filter Search",
                "categories": ["cs.CV"],
                "include_keywords": ["transformer"],
                "exclude_keywords": ["survey"],
                "author_filters": ["Jane Doe"],
                "date_window_days": 30,
                "min_citations": 10,
            },
            headers={"X-CSRF-Token": self.csrf_token},
        )
        self.assertEqual(create_response.status_code, 201)
        search_id = create_response.get_json()["id"]

        run_response = self.client.post(
            f"/api/saved-searches/{search_id}/run",
            json={"limit": 25},
            headers={"X-CSRF-Token": self.csrf_token},
        )

        self.assertEqual(run_response.status_code, 200)
        data = run_response.get_json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["results"][0]["title"], "Transformer Methods for Vision")
        self.assertEqual(data["results"][0]["authors"], "Jane Doe, Alex Smith")


class AuthorAutocompleteQaTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_author_autocomplete_orders_by_paper_count_then_name(self):
        db.session.add_all(
            [
                _make_paper(20, authors="Jane Doe, Ann Brown"),
                _make_paper(21, authors="Jane Doe, Anna Adams"),
                _make_paper(22, authors="Ann Brown"),
            ]
        )
        db.session.commit()

        response = self.client.get("/api/authors?q=Ann")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data[0], {"name": "Ann Brown", "paper_count": 2})
        self.assertEqual(data[1], {"name": "Anna Adams", "paper_count": 1})


if __name__ == "__main__":
    import unittest

    unittest.main()
