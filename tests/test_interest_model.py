"""Tests for the learned interest profile (feedback + embeddings)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from app.models import Collection, Paper, PaperCollection, PaperFeedback, db
from app.services.interest_model import (
    AFFINITY_Z_MIN,
    MIN_POSITIVE_FEEDBACK,
    affinity_scores,
    build_interest_profile,
    collection_affinity,
    fit_collection_profile,
    recompute_interest_similarities,
    reset_interest_profile_cache,
    score_vector,
)
from app.services.learned_ranker import interest_signal, resolve_interest_source
from app.services.metrics import feature_liveness
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

    def index_size(self) -> int:
        return len(self.vectors_by_id)

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

    def test_cache_invalidates_when_backlog_papers_get_indexed(self):
        # 5 saved papers but only 4 embedded at first → profile reads as disabled.
        # When the 5th embeds later (no new feedback), the feedback-only cache key
        # wouldn't change; keying on index size too must force the recompute.
        saved = self._add_feedback("save", MIN_POSITIVE_FEEDBACK, axis=0)
        service = self._fake_service({0: saved[:-1]})  # only 4 of 5 vectors indexed

        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            self.assertIsNone(build_interest_profile(self.app))

            # The 5th paper gets embedded by a later scrape — no new feedback.
            service.vectors_by_id[saved[-1].id] = _basis_vector(0)

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

    def test_collection_affinity_profile(self):
        """Collections alone are the interest model: centred centroids, z against the background."""
        rng = np.random.default_rng(0)
        shared = rng.normal(size=768)  # what every abstract has in common; centring removes it
        vectors: dict[int, np.ndarray] = {}

        def embed(topic=None) -> np.ndarray:
            vec = shared + (rng.normal(size=768) if topic is None else topic + 0.3 * rng.normal(size=768))
            return (vec / np.linalg.norm(vec)).astype(np.float32)

        def add(vector: np.ndarray, match_type: str = "Title") -> int:
            paper = _paper(f"2801.{10000 + len(vectors)}")
            paper.match_type = match_type
            db.session.add(paper)
            db.session.flush()
            vectors[paper.id] = vector
            return paper.id

        topics = {"Alpha": rng.normal(size=768), "Beta": rng.normal(size=768)}
        collections, members = {}, {}
        for name, topic in topics.items():
            collection = Collection(name=name)
            db.session.add(collection)
            db.session.flush()
            collections[name] = collection.id
            members[name] = [add(embed(topic)) for _ in range(MIN_POSITIVE_FEEDBACK)]  # just enough for a row
            db.session.add_all(PaperCollection(paper_id=pid, collection_id=collection.id) for pid in members[name])
        # A Beta-like paper excluded from Alpha: neither an Alpha member nor background.
        excluded = add(embed(topics["Beta"]))
        db.session.add(PaperCollection(paper_id=excluded, collection_id=collections["Alpha"], decision="exclude"))
        # Admitted by the gate itself and filed nowhere: kept out of the background too.
        add(embed(topics["Alpha"]), match_type="Interest")
        background = [add(embed()) for _ in range(250)]
        db.session.commit()
        service = _FakeEmbeddingService(vectors)

        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            profile = build_interest_profile(self.app)

            # The scorer, recomputed by hand from the spec.
            mean = np.asarray(list(vectors.values())).mean(axis=0)
            outside = np.asarray([vectors[pid] for pid in background])
            unrelated = outside - mean
            unrelated /= np.linalg.norm(unrelated, axis=1, keepdims=True)
            for name, collection_id in collections.items():
                row = profile.collection_ids.index(collection_id)
                centroid = (np.asarray([vectors[pid] for pid in members[name]]) - mean).mean(axis=0)
                centroid /= np.linalg.norm(centroid)
                np.testing.assert_allclose(profile.centroids[row], centroid, atol=1e-5)
                cosines = unrelated @ centroid
                np.testing.assert_allclose(profile.bg_mean[row], cosines.mean(), atol=1e-5)
                np.testing.assert_allclose(profile.bg_std[row], cosines.std(), atol=1e-5)
                self.assertEqual(profile.labels[row], name)
            # One member fewer than MIN_POSITIVE_FEEDBACK and a collection gets no row.
            too_few = np.asarray([vectors[pid] for pid in members["Alpha"][1:]])
            self.assertIsNone(fit_collection_profile({1: too_few}, outside, mean))

            held_out = embed(topics["Alpha"])
            z, collection_id = collection_affinity(profile, held_out)
            self.assertGreaterEqual(z, AFFINITY_Z_MIN)
            self.assertEqual(collection_id, collections["Alpha"])
            self.assertEqual(score_vector(profile, held_out), 1.0)  # z / 4, clipped
            # The best collection, not the first: the excluded paper is Beta-like.
            self.assertEqual(collection_affinity(profile, vectors[excluded])[1], collections["Beta"])

            randoms = np.asarray([embed() for _ in range(20)])
            scores = affinity_scores(profile, randoms)
            self.assertEqual(scores.shape, (20, 2))
            self.assertGreaterEqual((scores.max(axis=1) < AFFINITY_Z_MIN).mean(), 0.8)
            # Scoring centres the paper too, before the cosine and the z.
            centred = randoms - mean
            centred /= np.linalg.norm(centred, axis=1, keepdims=True)
            by_hand = (centred @ profile.centroids.T - profile.bg_mean) / profile.bg_std
            np.testing.assert_allclose(scores, by_hand, atol=1e-3)

            # The description (here: the paper itself, cosine 1) is not blended in: the signal
            # is the best z over 4 (AFFINITY_Z_SCALE), which makes the "> 0.5" explanation z > 2.
            far = randoms[int(scores.max(axis=1).argmin())]
            signal, source = interest_signal(far, profile, None, 0.7, description_vector=far)
            self.assertEqual(source, "collection")
            self.assertAlmostEqual(signal, float(scores.max(axis=1).min()) / 4.0, places=4)
            self.assertLess(signal, 0.5)

            client = self.app.test_client()
            self.assertIs(feature_liveness()["interest_signal_ready"], True)
            self.assertIn("Your collections rank papers", client.get("/settings").get_data(as_text=True))

            # A web process starts with an empty cache (the daily scrape is another
            # process): the pages that explain scores build the profile themselves.
            for url in ("/", f"/api/papers/{background[0]}/explain"):
                reset_interest_profile_cache()
                self.assertEqual(resolve_interest_source(None), "centroid")
                self.assertEqual(client.get(url).status_code, 200)
                self.assertEqual(resolve_interest_source(None), "collection", url)

            # Rescoring the stored papers needs no encoder: the description is not embedded.
            with patch("app.services.learned_ranker.active_description_vector") as embed_description:
                recompute_interest_similarities(self.app)
            embed_description.assert_not_called()
            db.session.expire_all()
            self.assertEqual(db.session.get(Paper, members["Alpha"][0]).interest_similarity, 1.0)

            fingerprints = [profile.fingerprint]

            def rebuilt():
                """The profile after the pending edit, which must have changed the fingerprint."""
                db.session.commit()
                fresh = build_interest_profile(self.app)
                self.assertNotIn(fresh.fingerprint, fingerprints)
                fingerprints.append(fresh.fingerprint)
                return fresh

            # A new membership changes the fingerprint, so the cached profile is rebuilt.
            db.session.add(PaperCollection(paper_id=background[0], collection_id=collections["Beta"]))
            rebuilt()
            # So does every edit that keeps the in-review count and id sums: a rename (the
            # gate stores the name on the papers it admits), two members swapped between the
            # collections, and an excluded row for a background paper.
            db.session.get(Collection, collections["Alpha"]).name = "Gamma"
            self.assertIn("Gamma", rebuilt().labels)
            for name, other in (("Alpha", "Beta"), ("Beta", "Alpha")):
                PaperCollection.query.filter_by(paper_id=members[name][0]).one().collection_id = collections[other]
            rebuilt()
            db.session.add(
                PaperCollection(paper_id=background[1], collection_id=collections["Alpha"], decision="exclude")
            )
            rebuilt()

            # A non-finite row in the index is left out instead of turning every z into NaN.
            add(np.full(768, np.nan, dtype=np.float32))
            self.assertTrue(np.isfinite(rebuilt().bg_std).all())

            # Enough ratings for a feedback profile change nothing while collections are the
            # model, and the Settings card keeps saying who ranks.
            saved = background[-MIN_POSITIVE_FEEDBACK:]
            db.session.add_all(PaperFeedback(paper_id=pid, action="save") for pid in saved)
            self.assertIsNotNone(rebuilt().centroids)
            status = feature_liveness()
            self.assertEqual((status["positive_feedback_needed"], status["collection_profile"]), (0, True))
            self.assertIn("Your collections rank papers", client.get("/settings").get_data(as_text=True))

            # Too small a background for z statistics: no collection profile. The saved
            # papers leave the index with it, so there is no feedback profile either.
            for pid in background[51:]:
                del service.vectors_by_id[pid]
            self.assertIsNone(build_interest_profile(self.app))


if __name__ == "__main__":
    unittest.main()
