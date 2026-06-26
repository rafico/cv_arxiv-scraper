"""Tests for ranking-onboarding: cold-start bootstrap + active-learning loop."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Import the route module at module load (before any create_app registers the
# blueprint) so its handlers attach to the shared api_bp. The production wiring
# adds `onboarding` to app/routes/api/__init__.py's import tuple.
import app.routes.api.onboarding  # noqa: F401
from app.models import Paper, PaperFeedback, db
from app.services.onboarding import (
    bootstrap_from_arxiv_ids,
    normalize_arxiv_id,
    select_uncertain_papers,
)
from tests.helpers import FlaskDBTestCase

# A canned 2-entry arXiv Atom feed (as returned by the API id_list query).
_ATOM_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>First Bootstrap Paper</title>
    <summary>An abstract about vision transformers and detection.</summary>
    <published>2024-01-02T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link href="http://arxiv.org/abs/2401.00001v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.00001v1" rel="related" type="application/pdf"/>
    <category term="cs.CV" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v2</id>
    <title>Second Bootstrap Paper</title>
    <summary>An abstract about reinforcement learning.</summary>
    <published>2024-01-03T00:00:00Z</published>
    <author><name>Carol King</name></author>
    <link href="http://arxiv.org/abs/2401.00002v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.00002v2" rel="related" type="application/pdf"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""


def _basis_vector(axis: int) -> np.ndarray:
    vec = np.zeros(768, dtype=np.float32)
    vec[axis] = 1.0
    return vec


class _FakeEmbeddingService:
    """Stub matching EmbeddingService.add_papers/get_paper_vectors/index_size."""

    def __init__(self, vectors_by_id: dict[int, np.ndarray] | None = None):
        self.vectors_by_id: dict[int, np.ndarray] = dict(vectors_by_id or {})
        self.added: list[int] = []

    def index_size(self) -> int:
        return len(self.vectors_by_id)

    def add_papers(self, paper_ids, texts, vectors=None) -> int:
        del texts, vectors
        count = 0
        for pid in paper_ids:
            if pid not in self.vectors_by_id:
                # Default to a fixed basis vector so a newly added paper is scorable.
                self.vectors_by_id[pid] = _basis_vector(0)
                self.added.append(pid)
                count += 1
        return count

    def get_paper_vectors(self, paper_ids):
        found = [pid for pid in paper_ids if pid in self.vectors_by_id]
        if not found:
            return [], np.empty((0, 768), dtype=np.float32)
        return found, np.asarray([self.vectors_by_id[pid] for pid in found], dtype=np.float32)


def _make_paper(arxiv_id: str, **overrides) -> Paper:
    defaults = dict(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        authors="Author A",
        link=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_link=f"https://arxiv.org/pdf/{arxiv_id}",
        abstract_text="abstract",
        categories=["cs.CV"],
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=1.0,
        publication_date="2026-01-01",
        scraped_date="2026-01-01",
    )
    defaults.update(overrides)
    return Paper(**defaults)


def _fake_response(content: bytes) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.text = content.decode("utf-8")
    return resp


class NormalizeArxivIdTests(unittest.TestCase):
    def test_bare_new_id(self):
        self.assertEqual(normalize_arxiv_id("2401.01234"), "2401.01234")

    def test_strips_version_suffix(self):
        self.assertEqual(normalize_arxiv_id("2401.01234v3"), "2401.01234")

    def test_strips_arxiv_prefix(self):
        self.assertEqual(normalize_arxiv_id("arXiv:2401.01234"), "2401.01234")

    def test_abs_url(self):
        self.assertEqual(normalize_arxiv_id("https://arxiv.org/abs/2401.01234v2"), "2401.01234")

    def test_pdf_url(self):
        self.assertEqual(normalize_arxiv_id("http://arxiv.org/pdf/2401.01234"), "2401.01234")

    def test_legacy_scheme(self):
        # The subject class (".GT") is metadata, not part of the canonical id the
        # arXiv API resolves, so it is stripped: math.GT/0309136 -> math/0309136.
        self.assertEqual(normalize_arxiv_id("arXiv:math.GT/0309136"), "math/0309136")

    def test_legacy_scheme_multichar_subcategory(self):
        # cond-mat.str-el used to corrupt to "str-el/0309136" (the greedy 2-letter
        # subclass match); the hyphenated archive must be preserved and the subclass
        # dropped.
        self.assertEqual(normalize_arxiv_id("cond-mat.str-el/0309136"), "cond-mat/0309136")
        self.assertEqual(normalize_arxiv_id("arXiv:q-bio.NC/0511032"), "q-bio/0511032")
        self.assertEqual(normalize_arxiv_id("https://arxiv.org/abs/math/0309136"), "math/0309136")

    def test_empty_and_garbage(self):
        self.assertIsNone(normalize_arxiv_id(""))
        self.assertIsNone(normalize_arxiv_id("   "))
        self.assertIsNone(normalize_arxiv_id("not-an-id"))
        self.assertIsNone(normalize_arxiv_id(None))  # type: ignore[arg-type]


class BootstrapTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.service = _FakeEmbeddingService()

    def _patches(self):
        return [
            patch("app.services.onboarding.request_with_backoff", return_value=_fake_response(_ATOM_FEED)),
            patch("app.services.onboarding.get_embedding_service", return_value=self.service),
            patch("app.services.embeddings.get_embedding_service", return_value=self.service),
        ]

    def test_bootstrap_inserts_new_papers_and_marks_saved(self):
        recompute = MagicMock(return_value=0)
        patches = self._patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            patch("app.services.interest_model.recompute_interest_similarities", recompute),
        ):
            summary = bootstrap_from_arxiv_ids(["2401.00001", "2401.00002"], app=self.app)

        self.assertEqual(summary["requested"], 2)
        self.assertCountEqual(summary["ingested"], ["2401.00001", "2401.00002"])
        self.assertEqual(summary["already_present"], [])
        self.assertEqual(summary["failed"], [])
        self.assertEqual(summary["saved_total"], 2)

        # Papers were inserted, deduped by arxiv_id.
        self.assertIsNotNone(Paper.query.filter_by(arxiv_id="2401.00001").one_or_none())
        self.assertIsNotNone(Paper.query.filter_by(arxiv_id="2401.00002").one_or_none())

        # Save feedback rows exist for both.
        save_rows = PaperFeedback.query.filter_by(action="save").all()
        self.assertEqual(len(save_rows), 2)

        # The recompute ran exactly once after the loop.
        recompute.assert_called_once()

    def test_bootstrap_dedupes_existing_arxiv_id(self):
        existing = _make_paper("2401.00001")
        db.session.add(existing)
        db.session.commit()
        existing_id = existing.id

        recompute = MagicMock(return_value=0)
        patches = self._patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            patch("app.services.interest_model.recompute_interest_similarities", recompute),
        ):
            summary = bootstrap_from_arxiv_ids(["arXiv:2401.00001v4", "2401.00002"], app=self.app)

        # No duplicate Paper for the existing arxiv_id.
        self.assertEqual(Paper.query.filter_by(arxiv_id="2401.00001").count(), 1)
        self.assertEqual(Paper.query.filter_by(arxiv_id="2401.00001").one().id, existing_id)

        self.assertEqual(summary["already_present"], ["2401.00001"])
        self.assertEqual(summary["ingested"], ["2401.00002"])
        self.assertEqual(summary["saved_total"], 2)

    def test_bootstrap_empty_returns_zero_summary(self):
        summary = bootstrap_from_arxiv_ids([], app=self.app)
        self.assertEqual(summary["requested"], 0)
        self.assertEqual(summary["ingested"], [])
        self.assertFalse(summary["profile_active"])

    def test_bootstrap_does_not_unsave_already_saved_paper(self):
        # A paper the user already saved must stay saved when bootstrapped again —
        # apply_feedback_action is a TOGGLE, so an unguarded call would unsave it.
        existing = _make_paper("2401.00001")
        db.session.add(existing)
        db.session.commit()
        existing_id = existing.id
        db.session.add(PaperFeedback(paper_id=existing_id, action="save"))
        db.session.commit()

        recompute = MagicMock(return_value=0)
        patches = self._patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            patch("app.services.interest_model.recompute_interest_similarities", recompute),
        ):
            summary = bootstrap_from_arxiv_ids(["2401.00001"], app=self.app)

        self.assertEqual(
            PaperFeedback.query.filter_by(paper_id=existing_id, action="save").count(),
            1,
        )
        self.assertEqual(summary["already_present"], ["2401.00001"])
        self.assertEqual(summary["failed"], [])
        self.assertEqual(summary["saved_total"], 1)


class SelectUncertainTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.service = _FakeEmbeddingService()

    def _save_paper(self, arxiv_id: str, vector: np.ndarray) -> Paper:
        paper = _make_paper(arxiv_id)
        db.session.add(paper)
        db.session.flush()
        db.session.add(PaperFeedback(paper_id=paper.id, action="save"))
        self.service.vectors_by_id[paper.id] = vector
        db.session.commit()
        return paper

    def _candidate_paper(self, arxiv_id: str, vector: np.ndarray) -> Paper:
        paper = _make_paper(arxiv_id)
        db.session.add(paper)
        db.session.flush()
        self.service.vectors_by_id[paper.id] = vector
        db.session.commit()
        return paper

    def test_returns_empty_below_min_saves(self):
        self._save_paper("2401.10001", _basis_vector(0))
        self._save_paper("2401.10002", _basis_vector(0))
        with patch("app.services.onboarding.get_embedding_service", return_value=self.service):
            self.assertEqual(select_uncertain_papers(limit=2, min_saves=3), [])

    def test_picks_mid_range_candidate(self):
        # Centroid is basis axis 0 (all saves point there).
        for i in range(3):
            self._save_paper(f"2401.2000{i}", _basis_vector(0))

        # Candidate similarities to the axis-0 centroid:
        #   high  ~ 1.0   (aligned)
        #   mid   ~ 0.5   (45-degree-ish blend)
        #   low   ~ 0.0   (orthogonal)
        # Range = [0, 1], midpoint = 0.5 → the "mid" candidate is most uncertain.
        high = _basis_vector(0)
        mid = (_basis_vector(0) + _basis_vector(1)).astype(np.float32)
        mid /= float(np.linalg.norm(mid))
        low = _basis_vector(1)

        self._candidate_paper("2401.30001", high)
        mid_paper = self._candidate_paper("2401.30002", mid)
        self._candidate_paper("2401.30003", low)

        with patch("app.services.onboarding.get_embedding_service", return_value=self.service):
            picks = select_uncertain_papers(limit=1, min_saves=3)

        self.assertEqual(len(picks), 1)
        self.assertEqual(picks[0]["paper_id"], mid_paper.id)
        self.assertIn("title", picks[0])
        self.assertIn("authors", picks[0])
        self.assertIn("similarity", picks[0])

    def test_excludes_papers_with_feedback(self):
        for i in range(3):
            self._save_paper(f"2401.4000{i}", _basis_vector(0))
        # A candidate that already has a skip must not be selected.
        skipped = self._candidate_paper("2401.50001", _basis_vector(1))
        db.session.add(PaperFeedback(paper_id=skipped.id, action="skip"))
        # A clean candidate that should be selected instead.
        clean = self._candidate_paper("2401.50002", _basis_vector(2))
        db.session.commit()

        with patch("app.services.onboarding.get_embedding_service", return_value=self.service):
            picks = select_uncertain_papers(limit=2, min_saves=3)

        picked_ids = {p["paper_id"] for p in picks}
        self.assertNotIn(skipped.id, picked_ids)
        self.assertIn(clean.id, picked_ids)


class OnboardingEndpointTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        self.service = _FakeEmbeddingService()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_bootstrap_requires_csrf(self):
        response = self.client.post("/api/onboarding/bootstrap", json={"arxiv_ids": ["2401.00001"]})
        self.assertEqual(response.status_code, 400)

    def test_bootstrap_endpoint_ingests(self):
        recompute = MagicMock(return_value=0)
        with (
            patch("app.services.onboarding.request_with_backoff", return_value=_fake_response(_ATOM_FEED)),
            patch("app.services.onboarding.get_embedding_service", return_value=self.service),
            patch("app.services.embeddings.get_embedding_service", return_value=self.service),
            patch("app.services.interest_model.recompute_interest_similarities", recompute),
        ):
            response = self.client.post(
                "/api/onboarding/bootstrap",
                json={"arxiv_ids": "2401.00001, 2401.00002"},
                headers={"X-CSRF-Token": self._csrf_token()},
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["saved_total"], 2)
        self.assertCountEqual(data["ingested"], ["2401.00001", "2401.00002"])

    def test_bootstrap_endpoint_rejects_empty(self):
        response = self.client.post(
            "/api/onboarding/bootstrap",
            json={"arxiv_ids": []},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(response.status_code, 400)

    def test_uncertain_endpoint_below_min_saves(self):
        with patch("app.services.onboarding.get_embedding_service", return_value=self.service):
            response = self.client.get("/api/onboarding/uncertain?limit=2")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["papers"], [])
        self.assertEqual(data["min_saves"], 3)
        self.assertEqual(data["saved_total"], 0)

    def test_uncertain_endpoint_returns_papers(self):
        # 3 saved papers (centroid on axis 0) + one clean orthogonal candidate.
        for i in range(3):
            paper = _make_paper(f"2401.6000{i}")
            db.session.add(paper)
            db.session.flush()
            db.session.add(PaperFeedback(paper_id=paper.id, action="save"))
            self.service.vectors_by_id[paper.id] = _basis_vector(0)
        candidate = _make_paper("2401.70001")
        db.session.add(candidate)
        db.session.flush()
        self.service.vectors_by_id[candidate.id] = _basis_vector(1)
        db.session.commit()

        with patch("app.services.onboarding.get_embedding_service", return_value=self.service):
            response = self.client.get("/api/onboarding/uncertain?limit=1")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["saved_total"], 3)
        self.assertEqual(len(data["papers"]), 1)
        entry = data["papers"][0]
        self.assertEqual(entry["paper_id"], candidate.id)
        self.assertIn("title", entry)
        self.assertIn("authors", entry)
        self.assertIsInstance(entry["similarity"], float)


if __name__ == "__main__":
    unittest.main()
