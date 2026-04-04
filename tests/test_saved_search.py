"""Tests for SavedSearch execution engine and API endpoints."""

from __future__ import annotations

from datetime import date, timedelta

from app.models import Paper, SavedSearch, db
from app.services.saved_search import execute_saved_search, validate_saved_search
from app.services.text import now_utc
from tests.helpers import FlaskDBTestCase


class TestValidateSavedSearch:
    def test_valid_data(self):
        assert (
            validate_saved_search(
                {
                    "include_keywords": ["transformer", "attention"],
                    "date_window_days": 30,
                    "min_citations": 10,
                }
            )
            == []
        )

    def test_negative_date_window(self):
        errors = validate_saved_search({"date_window_days": -1})
        assert any("non-negative" in e for e in errors)

    def test_invalid_keywords_type(self):
        errors = validate_saved_search({"include_keywords": "not a list"})
        assert any("list" in e for e in errors)

    def test_invalid_keyword_items(self):
        errors = validate_saved_search({"include_keywords": [1, 2]})
        assert any("strings" in e for e in errors)


class TestExecuteSavedSearch(FlaskDBTestCase):
    def _add_paper(self, **kwargs):
        defaults = {
            "title": "Test Paper",
            "authors": "Author A",
            "link": f"https://arxiv.org/abs/{kwargs.get('arxiv_id', '0001')}",
            "pdf_link": f"https://arxiv.org/pdf/{kwargs.get('arxiv_id', '0001')}",
            "abstract_text": "An abstract about transformers.",
            "summary_text": "A summary.",
            "match_type": "Title",
            "matched_terms": ["transformer"],
            "paper_score": 10.0,
            "publication_date": "2026-04-01",
            "publication_dt": date(2026, 4, 1),
            "scraped_date": "2026-04-01",
            "scraped_at": now_utc(),
        }
        defaults.update(kwargs)
        paper = Paper(**defaults)
        db.session.add(paper)
        db.session.commit()
        return paper

    def test_include_keywords_filter(self):
        p1 = self._add_paper(title="Transformer Architecture", arxiv_id="k1", link="https://arxiv.org/abs/k1")
        p2 = self._add_paper(
            title="CNN Baseline", arxiv_id="k2", link="https://arxiv.org/abs/k2", abstract_text="A CNN paper."
        )
        search = SavedSearch(name="test", include_keywords=["transformer"])
        db.session.add(search)
        db.session.commit()
        results = execute_saved_search(search)
        result_ids = [p.id for p in results]
        assert p1.id in result_ids
        assert p2.id not in result_ids

    def test_exclude_keywords_filter(self):
        p1 = self._add_paper(
            title="Good Paper", arxiv_id="e1", link="https://arxiv.org/abs/e1", abstract_text="About vision."
        )
        p2 = self._add_paper(
            title="Bad Survey Paper",
            arxiv_id="e2",
            link="https://arxiv.org/abs/e2",
            abstract_text="A survey of methods.",
        )
        search = SavedSearch(name="test", exclude_keywords=["survey"])
        db.session.add(search)
        db.session.commit()
        results = execute_saved_search(search)
        result_ids = [p.id for p in results]
        assert p1.id in result_ids
        assert p2.id not in result_ids

    def test_author_filter(self):
        p1 = self._add_paper(
            title="P1", authors="Andrew Y. Ng, John Smith", arxiv_id="a1", link="https://arxiv.org/abs/a1"
        )
        p2 = self._add_paper(title="P2", authors="Jane Doe", arxiv_id="a2", link="https://arxiv.org/abs/a2")
        search = SavedSearch(name="test", author_filters=["Andrew Y. Ng"])
        db.session.add(search)
        db.session.commit()
        results = execute_saved_search(search)
        result_ids = [p.id for p in results]
        assert p1.id in result_ids
        assert p2.id not in result_ids

    def test_min_citations_filter(self):
        p1 = self._add_paper(title="Cited", arxiv_id="c1", link="https://arxiv.org/abs/c1", citation_count=50)
        p2 = self._add_paper(title="Uncited", arxiv_id="c2", link="https://arxiv.org/abs/c2", citation_count=2)
        search = SavedSearch(name="test", min_citations=10)
        db.session.add(search)
        db.session.commit()
        results = execute_saved_search(search)
        result_ids = [p.id for p in results]
        assert p1.id in result_ids
        assert p2.id not in result_ids

    def test_date_window_filter(self):
        recent = self._add_paper(
            title="Recent", arxiv_id="d1", link="https://arxiv.org/abs/d1", publication_dt=date.today()
        )
        old = self._add_paper(
            title="Old",
            arxiv_id="d2",
            link="https://arxiv.org/abs/d2",
            publication_dt=date.today() - timedelta(days=60),
        )
        search = SavedSearch(name="test", date_window_days=30)
        db.session.add(search)
        db.session.commit()
        results = execute_saved_search(search)
        result_ids = [p.id for p in results]
        assert recent.id in result_ids
        assert old.id not in result_ids

    def test_combined_filters(self):
        p1 = self._add_paper(
            title="Transformer for Remote Sensing",
            authors="Andrew Y. Ng",
            arxiv_id="combo1",
            link="https://arxiv.org/abs/combo1",
            citation_count=20,
        )
        self._add_paper(
            title="CNN Baseline",
            authors="Jane Doe",
            arxiv_id="combo2",
            link="https://arxiv.org/abs/combo2",
            citation_count=5,
        )
        search = SavedSearch(
            name="test",
            include_keywords=["transformer"],
            author_filters=["Andrew Y. Ng"],
            min_citations=10,
        )
        db.session.add(search)
        db.session.commit()
        results = execute_saved_search(search)
        assert len(results) == 1
        assert results[0].id == p1.id


class TestSavedSearchApi(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        # Initialize CSRF token.
        self.client.get("/")
        with self.client.session_transaction() as sess:
            self.csrf_token = sess["settings_csrf_token"]

    def test_create_saved_search_with_fields(self):
        response = self.client.post(
            "/api/saved-searches",
            json={
                "name": "My Search",
                "include_keywords": ["transformer"],
                "exclude_keywords": ["survey"],
                "date_window_days": 30,
            },
            headers={"X-CSRF-Token": self.csrf_token},
        )
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertEqual(data["name"], "My Search")
        self.assertEqual(data["include_keywords"], ["transformer"])
        self.assertEqual(data["exclude_keywords"], ["survey"])
        self.assertEqual(data["date_window_days"], 30)

    def test_get_saved_search(self):
        s = SavedSearch(name="Test", include_keywords=["nerf"])
        db.session.add(s)
        db.session.commit()
        response = self.client.get(f"/api/saved-searches/{s.id}")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["name"], "Test")
        self.assertEqual(data["include_keywords"], ["nerf"])

    def test_update_saved_search(self):
        s = SavedSearch(name="Old Name")
        db.session.add(s)
        db.session.commit()
        response = self.client.put(
            f"/api/saved-searches/{s.id}",
            json={"name": "New Name", "min_citations": 5},
            headers={"X-CSRF-Token": self.csrf_token},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["name"], "New Name")
        self.assertEqual(data["min_citations"], 5)

    def test_run_saved_search(self):
        paper = Paper(
            title="Transformer Paper",
            authors="Author A",
            link="https://arxiv.org/abs/run1",
            pdf_link="https://arxiv.org/pdf/run1",
            abstract_text="A paper about transformers.",
            summary_text="Summary.",
            match_type="Title",
            paper_score=10.0,
            publication_date="2026-04-01",
            scraped_date="2026-04-01",
            scraped_at=now_utc(),
        )
        db.session.add(paper)
        s = SavedSearch(name="Test", include_keywords=["transformer"])
        db.session.add(s)
        db.session.commit()

        response = self.client.post(
            f"/api/saved-searches/{s.id}/run",
            json={},
            headers={"X-CSRF-Token": self.csrf_token},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["results"][0]["title"], "Transformer Paper")

        # Running a search should update last_used_at.
        db.session.refresh(s)
        self.assertIsNotNone(s.last_used_at)

    def test_validate_rejects_bad_data(self):
        response = self.client.post(
            "/api/saved-searches",
            json={
                "name": "Bad",
                "date_window_days": -5,
            },
            headers={"X-CSRF-Token": self.csrf_token},
        )
        self.assertEqual(response.status_code, 400)
