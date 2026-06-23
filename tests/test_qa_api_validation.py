"""QA round 2 — API input-type validation.

Mutating JSON endpoints previously turned wrong-typed bodies into opaque 500s
(``.strip()`` on a non-str, set-membership on a non-hashable, iterating a non-list,
an unbounded IN list, a duplicate-URL IntegrityError). Each must now return a clean
4xx. A legacy saved-search row with a non-string ``filters.q`` must run, not 500.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from app.models import Collection, FeedSource, Paper, SavedSearch, db
from tests.helpers import FlaskDBTestCase


def _make_paper(idx: int = 0, **overrides) -> Paper:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    defaults = dict(
        arxiv_id=f"2607.{4000 + idx:04d}",
        title=f"Validation Paper {idx}",
        authors="Author A",
        link=f"https://arxiv.org/abs/2607.{4000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2607.{4000 + idx:04d}",
        abstract_text="abstract",
        summary_text="summary",
        match_type="Title",
        paper_score=10.0,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=now,
    )
    defaults.update(overrides)
    return Paper(**defaults)


class ApiInputValidationTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as sess:
            return sess["settings_csrf_token"]

    def _hdr(self) -> dict:
        return {"X-CSRF-Token": self._csrf()}

    # --- collections ------------------------------------------------------
    def test_create_collection_non_string_name(self):
        r = self.client.post("/api/collections", json={"name": 5}, headers=self._hdr())
        self.assertEqual(r.status_code, 400)

    def test_create_collection_non_string_description(self):
        r = self.client.post("/api/collections", json={"name": "ok", "description": {"x": 1}}, headers=self._hdr())
        self.assertEqual(r.status_code, 400)

    def test_add_papers_non_list_paper_ids(self):
        c = Collection(name="C1")
        db.session.add(c)
        db.session.commit()
        r = self.client.post(f"/api/collections/{c.id}/papers", json={"paper_ids": "1,2,3"}, headers=self._hdr())
        self.assertEqual(r.status_code, 400)

    def test_add_papers_skips_non_int_ids(self):
        # A list of stringy ids must not coerce / crash; valid ints still added.
        c = Collection(name="C2")
        p = _make_paper(1)
        db.session.add_all([c, p])
        db.session.commit()
        r = self.client.post(f"/api/collections/{c.id}/papers", json={"paper_ids": [p.id, "x"]}, headers=self._hdr())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["added"], 1)

    # --- papers -----------------------------------------------------------
    def test_reading_status_non_hashable(self):
        p = _make_paper(2)
        db.session.add(p)
        db.session.commit()
        r = self.client.post(f"/api/papers/{p.id}/reading-status", json={"status": {"bad": 1}}, headers=self._hdr())
        self.assertEqual(r.status_code, 400)

    def test_add_tag_non_string(self):
        p = _make_paper(3)
        db.session.add(p)
        db.session.commit()
        r = self.client.post(f"/api/papers/{p.id}/tags", json={"tag": 123}, headers=self._hdr())
        self.assertEqual(r.status_code, 400)

    # --- feed sources -----------------------------------------------------
    def test_feed_source_duplicate_url_returns_409(self):
        db.session.add(FeedSource(name="A", url="https://rss.arxiv.org/rss/cs.CV"))
        db.session.commit()
        r = self.client.post(
            "/api/feed-sources",
            json={"name": "B", "url": "https://rss.arxiv.org/rss/cs.CV"},
            headers=self._hdr(),
        )
        self.assertEqual(r.status_code, 409)

    def test_feed_source_non_string_url(self):
        r = self.client.post("/api/feed-sources", json={"name": "B", "url": 5}, headers=self._hdr())
        self.assertEqual(r.status_code, 400)

    # --- export -----------------------------------------------------------
    def test_bulk_bibtex_rejects_too_many_ids(self):
        ids = ",".join(str(i) for i in range(1, 1502))
        r = self.client.get(f"/api/papers/bulk-bibtex?ids={ids}")
        self.assertEqual(r.status_code, 400)

    # --- saved search -----------------------------------------------------
    def test_run_saved_search_with_non_string_legacy_q(self):
        # A legacy/hand-edited row may carry a non-string q; /run must not 500.
        s = SavedSearch(name="legacy", filters={"q": 123})
        db.session.add(s)
        db.session.commit()
        r = self.client.post(f"/api/saved-searches/{s.id}/run", json={}, headers=self._hdr())
        self.assertEqual(r.status_code, 200)

    def test_create_saved_search_rejects_non_string_q(self):
        r = self.client.post(
            "/api/saved-searches",
            json={"name": "x", "filters": {"q": 123}},
            headers=self._hdr(),
        )
        self.assertEqual(r.status_code, 400)

    # --- global error handler safety net ---------------------------------
    def test_top_level_json_array_body_is_400_not_500(self):
        r = self.client.post("/api/collections", json=[1, 2, 3], headers=self._hdr())
        self.assertEqual(r.status_code, 400)
