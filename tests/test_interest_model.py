"""Tests for the learned interest profile (feedback + embeddings)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from app.models import Paper, PaperFeedback, db
from app.services.interest_model import (
    MIN_POSITIVE_FEEDBACK,
    build_interest_profile,
    recompute_interest_similarities,
    reset_interest_profile_cache,
    score_vector,
)
from tests.helpers import FlaskDBTestCase


def _paper(arxiv_id: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        authors="Author A",
        link=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_link=f"https://arxiv.org/pdf/{arxiv_id}",
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=1.0,
        publication_date="2026-01-01",
        scraped_date="2026-01-01",
    )


def _basis_vector(axis: int) -> np.ndarray:
    vec = np.zeros(768, dtype=np.float32)
    vec[axis] = 1.0
    return vec


class _FakeEmbeddingService:
    """Stub matching the EmbeddingService.get_paper_vectors contract."""

    def __init__(self, vectors_by_id: dict[int, np.ndarray]):
        self.vectors_by_id = vectors_by_id

    def get_paper_vectors(self, paper_ids):
        found = [pid for pid in paper_ids if pid in self.vectors_by_id]
        if not found:
            return [], np.empty((0, 768), dtype=np.float32)
        return found, np.asarray([self.vectors_by_id[pid] for pid in found], dtype=np.float32)


class InterestProfileTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        reset_interest_profile_cache()

    def tearDown(self):
        reset_interest_profile_cache()
        super().tearDown()

    def _add_feedback(self, action: str, count: int, axis: int, start: int = 0) -> list[Paper]:
        papers = []
        for idx in range(count):
            paper = _paper(f"27{axis:02d}.{10000 + start + idx}")
            db.session.add(paper)
            db.session.flush()
            db.session.add(PaperFeedback(paper_id=paper.id, action=action))
            papers.append(paper)
        db.session.commit()
        return papers

    def _fake_service(self, papers_by_axis: dict[int, list[Paper]]) -> _FakeEmbeddingService:
        vectors = {}
        for axis, papers in papers_by_axis.items():
            for paper in papers:
                vectors[paper.id] = _basis_vector(axis)
        return _FakeEmbeddingService(vectors)

    def test_cold_start_without_feedback_returns_none(self):
        with patch("app.services.embeddings.get_embedding_service", return_value=_FakeEmbeddingService({})):
            self.assertIsNone(build_interest_profile(self.app))

    def test_below_threshold_returns_none(self):
        saved = self._add_feedback("save", MIN_POSITIVE_FEEDBACK - 1, axis=0)
        service = self._fake_service({0: saved})
        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            self.assertIsNone(build_interest_profile(self.app))

    def test_positive_profile_scores_similar_papers_high(self):
        saved = self._add_feedback("save", MIN_POSITIVE_FEEDBACK, axis=0)
        service = self._fake_service({0: saved})

        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            profile = build_interest_profile(self.app)

        self.assertIsNotNone(profile)
        self.assertIsNone(profile.neg_centroid)
        self.assertAlmostEqual(score_vector(profile, _basis_vector(0)), 1.0, places=5)
        self.assertAlmostEqual(score_vector(profile, _basis_vector(5)), 0.0, places=5)

    def test_negative_centroid_demotes_skipped_topics(self):
        saved = self._add_feedback("save", MIN_POSITIVE_FEEDBACK, axis=0)
        skipped = self._add_feedback("skip", 3, axis=1, start=100)
        service = self._fake_service({0: saved, 1: skipped})

        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            profile = build_interest_profile(self.app)

        self.assertIsNotNone(profile.neg_centroid)
        self.assertAlmostEqual(score_vector(profile, _basis_vector(0)), 1.0, places=5)
        self.assertAlmostEqual(score_vector(profile, _basis_vector(1)), -1.0, places=5)

    def test_fingerprint_cache_invalidates_on_new_feedback(self):
        service = self._fake_service({})
        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            self.assertIsNone(build_interest_profile(self.app))

            saved = self._add_feedback("save", MIN_POSITIVE_FEEDBACK, axis=0)
            for paper in saved:
                service.vectors_by_id[paper.id] = _basis_vector(0)

            self.assertIsNotNone(build_interest_profile(self.app))

    def test_score_vector_clamps_and_handles_zero_vector(self):
        saved = self._add_feedback("save", MIN_POSITIVE_FEEDBACK, axis=0)
        service = self._fake_service({0: saved})
        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            profile = build_interest_profile(self.app)

        self.assertEqual(score_vector(profile, np.zeros(768, dtype=np.float32)), 0.0)
        self.assertLessEqual(score_vector(profile, _basis_vector(0) * 7.5), 1.0)

    def test_recompute_interest_similarities_writes_column(self):
        saved = self._add_feedback("save", MIN_POSITIVE_FEEDBACK, axis=0)
        other = _paper("2799.99999")
        db.session.add(other)
        db.session.commit()
        service = self._fake_service({0: saved})
        service.vectors_by_id[other.id] = _basis_vector(0)

        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            updated = recompute_interest_similarities(self.app)

        self.assertGreaterEqual(updated, MIN_POSITIVE_FEEDBACK + 1)
        db.session.expire_all()
        stored = Paper.query.filter_by(arxiv_id="2799.99999").one()
        self.assertAlmostEqual(stored.interest_similarity, 1.0, places=3)
        self.assertGreater(stored.paper_score, 1.0)

    def test_recompute_clears_similarities_when_profile_gone(self):
        paper = _paper("2798.88888")
        paper.interest_similarity = 0.9
        db.session.add(paper)
        db.session.commit()

        with patch("app.services.embeddings.get_embedding_service", return_value=_FakeEmbeddingService({})):
            recompute_interest_similarities(self.app)

        db.session.expire_all()
        stored = Paper.query.filter_by(arxiv_id="2798.88888").one()
        self.assertIsNone(stored.interest_similarity)


if __name__ == "__main__":
    unittest.main()
