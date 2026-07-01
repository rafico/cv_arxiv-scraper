"""QA round 5 regression tests — API input-type hardening (seed defects).

- collections add_paper_to_collection: a JSON boolean ``paper_id``/``paper_ids``
  must be ignored, not coerced to Paper id 1/0 (``bool`` ⊂ ``int``). Mirrors the
  bulk_feedback guard.
- saved-search create/update: a non-string ``name`` must yield a *specific* 400
  ("'name' must be a string"), not an AttributeError that falls through to the
  generic safety-net "Invalid request" 400 + a logged traceback.
- historical scrape: a non-string ``start_date``/``end_date`` must yield the
  specific "Dates must be in YYYY-MM-DD format" 400 (strptime raises TypeError,
  not ValueError), not the generic safety-net 400.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from app.models import Paper, PaperCollection, SavedSearch, db
from tests.helpers import FlaskDBTestCase


def _make_paper(idx: int = 0, **overrides) -> Paper:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    defaults = dict(
        arxiv_id=f"2607.{3000 + idx:04d}",
        title=f"Round5 Test Paper {idx}",
        authors="Author A",
        link=f"https://arxiv.org/abs/2607.{3000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2607.{3000 + idx:04d}",
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


class _ApiTestCase(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]


class CollectionBooleanPaperIdTests(_ApiTestCase):
    def _make_collection(self) -> int:
        resp = self.client.post(
            "/api/collections",
            json={"name": "Box"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(resp.status_code, 201)
        return resp.get_json()["id"]

    def test_boolean_single_paper_id_is_ignored_not_paper_1(self):
        # The first inserted paper gets id 1; True == 1 would resolve to it pre-fix.
        db.session.add(_make_paper(0))
        db.session.commit()
        cid = self._make_collection()

        resp = self.client.post(
            f"/api/collections/{cid}/papers",
            json={"paper_id": True},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["added"], 0)
        self.assertEqual(PaperCollection.query.filter_by(collection_id=cid).count(), 0)

    def test_boolean_in_paper_ids_list_is_skipped(self):
        db.session.add(_make_paper(0))
        db.session.commit()
        cid = self._make_collection()

        resp = self.client.post(
            f"/api/collections/{cid}/papers",
            json={"paper_ids": [True]},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["added"], 0)
        self.assertEqual(PaperCollection.query.filter_by(collection_id=cid).count(), 0)

    def test_valid_integer_paper_id_still_added(self):
        paper = _make_paper(0)
        db.session.add(paper)
        db.session.commit()
        pid = paper.id
        cid = self._make_collection()

        resp = self.client.post(
            f"/api/collections/{cid}/papers",
            json={"paper_ids": [pid]},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["added"], 1)
        self.assertEqual(PaperCollection.query.filter_by(collection_id=cid).count(), 1)


class SavedSearchNonStringNameTests(_ApiTestCase):
    def test_create_non_string_name_returns_specific_400(self):
        resp = self.client.post(
            "/api/saved-searches",
            json={"name": 123, "filters": {}},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"], "'name' must be a string")

    def test_update_non_string_name_returns_specific_400(self):
        s = SavedSearch(name="Original", filters={})
        db.session.add(s)
        db.session.commit()
        sid = s.id

        resp = self.client.put(
            f"/api/saved-searches/{sid}",
            json={"name": ["nope"]},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"], "'name' must be a string")
        # The original name is untouched.
        self.assertEqual(db.session.get(SavedSearch, sid).name, "Original")

    def test_create_valid_name_still_works(self):
        resp = self.client.post(
            "/api/saved-searches",
            json={"name": "Good Name", "filters": {}},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.get_json()["name"], "Good Name")


class SavedSearchIntCoercionTests(_ApiTestCase):
    def test_int_parseable_string_persists_as_int_and_run_succeeds(self):
        # int("1_0") == 10 passes validation, but SQLite's INTEGER affinity does NOT
        # coerce "1_0" — it would persist as TEXT and later crash timedelta(days=...)
        # on /run. The route must store the coerced int.
        create = self.client.post(
            "/api/saved-searches",
            json={"name": "underscored", "date_window_days": "1_0"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(create.status_code, 201)
        sid = create.get_json()["id"]

        stored = db.session.get(SavedSearch, sid)
        self.assertIsInstance(stored.date_window_days, int)
        self.assertEqual(stored.date_window_days, 10)

        run = self.client.post(
            f"/api/saved-searches/{sid}/run",
            json={},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(run.status_code, 200)

    def test_update_int_parseable_string_persists_as_int(self):
        s = SavedSearch(name="Original", filters={})
        db.session.add(s)
        db.session.commit()
        sid = s.id

        resp = self.client.put(
            f"/api/saved-searches/{sid}",
            json={"min_citations": "1_0"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(resp.status_code, 200)
        stored = db.session.get(SavedSearch, sid)
        self.assertIsInstance(stored.min_citations, int)
        self.assertEqual(stored.min_citations, 10)


class HistoricalScrapeNonStringDateTests(_ApiTestCase):
    def test_non_string_start_date_returns_format_error(self):
        resp = self.client.post(
            "/api/search/historical",
            json={"categories": ["cs.CV"], "start_date": 123, "end_date": "2026-01-02"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"], "Dates must be in YYYY-MM-DD format")


if __name__ == "__main__":
    import unittest

    unittest.main()
