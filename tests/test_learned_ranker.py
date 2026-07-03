"""Tests for the learned ranker (Scholar Inbox recipe) + dense-retrieval candidates."""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.models import Paper, PaperFeedback, RecommendationMetric, db
from app.services import learned_ranker
from app.services.interest_model import (
    InterestProfile,
    reset_interest_profile_cache,
)
from app.services.learned_ranker import (
    LearnedModel,
    evaluate_learned_ranker,
    flush_pending_retrain,
    interest_signal,
    model_status,
    peek_learned_model,
    request_retrain,
    reset_learned_ranker_cache,
    resolve_interest_source,
    score_vectors,
    set_runtime_learned_prefs,
    train_learned_ranker,
)
from app.services.pipeline import WhitelistCandidateGenerator
from app.services.ranking import (
    explain_score,
    generate_ranking_explanation,
    top_score_contributors,
)
from tests.helpers import FlaskDBTestCase

DIM = 768


def _unit(axis: int) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float32)
    vec[axis] = 1.0
    return vec


def _noisy_unit(axis: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = _unit(axis) + rng.normal(0, 0.05, DIM).astype(np.float32)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def _random_unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + seed)
    vec = rng.normal(0, 1, DIM).astype(np.float32)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


class FakeEmbeddingService:
    """Stub matching the EmbeddingService methods the learned ranker uses."""

    def __init__(self, vectors_by_id: dict[int, np.ndarray]):
        self.vectors_by_id = vectors_by_id

    def index_size(self) -> int:
        return len(self.vectors_by_id)

    def get_paper_vectors(self, paper_ids):
        found = [pid for pid in paper_ids if pid in self.vectors_by_id]
        if not found:
            return [], np.empty((0, DIM), dtype=np.float32)
        return found, np.asarray([self.vectors_by_id[pid] for pid in found], dtype=np.float32)

    def sample_paper_vectors(self, count, exclude_ids=None, seed=None):
        exclude = exclude_ids or set()
        ids = sorted(pid for pid in self.vectors_by_id if pid not in exclude)[:count]
        if not ids:
            return [], np.empty((0, DIM), dtype=np.float32)
        return ids, np.asarray([self.vectors_by_id[pid] for pid in ids], dtype=np.float32)


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
        publication_date="2026-06-01",
        scraped_date="2026-06-01",
    )


class LearnedRankerTestCase(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        reset_learned_ranker_cache()
        reset_interest_profile_cache()
        # Unique per-test dir: the sandboxed instance path is session-wide, so a
        # shared path would leak the trained artifact between tests.
        import shutil
        import tempfile

        self._faiss_dir = tempfile.mkdtemp(prefix="learned_ranker_faiss_")
        self.addCleanup(shutil.rmtree, self._faiss_dir, True)
        self.app.config["FAISS_INDEX_DIR"] = self._faiss_dir

    def tearDown(self):
        reset_learned_ranker_cache()
        reset_interest_profile_cache()
        super().tearDown()

    def _add_labeled(self, labels: list[int]) -> tuple[dict[int, np.ndarray], list[Paper]]:
        """Create papers + feedback rows following `labels` (1=save, 0=skip).

        Timestamps increase per item so the eval's time-ordered split is
        deterministic. Positive papers embed near axis 0, negatives near axis 1.
        """
        vectors: dict[int, np.ndarray] = {}
        papers: list[Paper] = []
        base = datetime(2026, 6, 1, 8, 0, 0)
        for idx, label in enumerate(labels):
            paper = _paper(f"2606.{20000 + idx}")
            db.session.add(paper)
            db.session.flush()
            action = "save" if label else "skip"
            db.session.add(
                PaperFeedback(paper_id=paper.id, action=action, created_at=base + timedelta(hours=idx))
            )
            vectors[paper.id] = _noisy_unit(0 if label else 1, seed=idx)
            papers.append(paper)
        db.session.commit()
        return vectors, papers

    def _fake_service(self, labeled_vectors: dict[int, np.ndarray], corpus_count: int = 40) -> FakeEmbeddingService:
        vectors = dict(labeled_vectors)
        next_id = (max(vectors) if vectors else 0) + 1000
        for idx in range(corpus_count):
            vectors[next_id + idx] = _random_unit(idx)
        return FakeEmbeddingService(vectors)


class RecipeTrainingTests(LearnedRankerTestCase):
    """The recipe trains on synthetic separable data and scores sanely."""

    def test_trains_and_separates_positive_from_negative(self):
        labeled, _ = self._add_labeled([1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        service = self._fake_service(labeled)

        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            status = train_learned_ranker(self.app)

        self.assertTrue(status["available"])
        self.assertEqual(status["n_positive"], 6)
        self.assertEqual(status["n_negative"], 4)

        probs = score_vectors(np.asarray([_unit(0), _unit(1)]))
        self.assertIsNotNone(probs)
        self.assertGreater(float(probs[0]), 0.6)
        self.assertLess(float(probs[1]), 0.4)
        # A random off-topic vector should not look like an interest hit.
        random_prob = float(score_vectors(np.asarray([_random_unit(999)]))[0])
        self.assertLess(random_prob, float(probs[0]))

    def test_artifact_persists_and_reloads_across_cache_reset(self):
        labeled, _ = self._add_labeled([1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        service = self._fake_service(labeled)
        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            train_learned_ranker(self.app)

        artifact = Path(self._faiss_dir) / "learned_ranker.npz"
        self.assertTrue(artifact.is_file())

        before = float(score_vectors(np.asarray([_unit(0)]))[0])
        reset_learned_ranker_cache()
        model = peek_learned_model()  # loads from disk, no DB / training
        self.assertIsNotNone(model)
        after = float(model.predict_proba(np.asarray([_unit(0)]))[0])
        self.assertAlmostEqual(before, after, places=5)

    def test_retrain_is_fingerprint_cached(self):
        labeled, _ = self._add_labeled([1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        service = self._fake_service(labeled)
        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            first = train_learned_ranker(self.app)
            second = train_learned_ranker(self.app)
        self.assertEqual(first["reason"], "trained")
        self.assertEqual(second["reason"], "cached")

    def test_evaluation_writes_metric_rows(self):
        labeled, _ = self._add_labeled([1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        service = self._fake_service(labeled)
        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            result = evaluate_learned_ranker(self.app)

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["auc"], 0.5)
        auc_rows = RecommendationMetric.query.filter_by(metric_name="learned_ranker_auc").all()
        f1_rows = RecommendationMetric.query.filter_by(metric_name="learned_ranker_f1").all()
        self.assertEqual(len(auc_rows), 1)
        self.assertEqual(len(f1_rows), 1)

    def test_numpy_fallback_matches_recipe_behavior(self):
        """The pure-numpy IRLS path (no sklearn) still separates the classes."""
        labeled, _ = self._add_labeled([1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        service = self._fake_service(labeled)

        import builtins

        real_import = builtins.__import__

        def _no_sklearn(name, *args, **kwargs):
            if name.startswith("sklearn"):
                raise ImportError("sklearn disabled for test")
            return real_import(name, *args, **kwargs)

        with (
            patch("app.services.embeddings.get_embedding_service", return_value=service),
            patch("builtins.__import__", side_effect=_no_sklearn),
        ):
            status = train_learned_ranker(self.app)

        self.assertTrue(status["available"])
        probs = score_vectors(np.asarray([_unit(0), _unit(1)]))
        self.assertGreater(float(probs[0]), 0.6)
        self.assertLess(float(probs[1]), 0.4)


class ColdStartTests(LearnedRankerTestCase):
    def test_below_minimum_positive_feedback_reports_unavailable(self):
        labeled, _ = self._add_labeled([1, 1, 1, 0])  # 3 saves < MIN_POSITIVE_FEEDBACK
        service = self._fake_service(labeled)
        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            status = train_learned_ranker(self.app)

        self.assertFalse(status["available"])
        self.assertEqual(status["reason"], "not-enough-feedback")
        self.assertGreater(status["needed_positive"], 0)
        self.assertIsNone(peek_learned_model())
        self.assertIsNone(score_vectors(np.asarray([_unit(0)])))

    def test_interest_signal_falls_back_to_centroid(self):
        profile = InterestProfile(pos_centroid=_unit(0), neg_centroid=None, fingerprint=(1, 1, 1))
        signal, source = interest_signal(_unit(0), profile, None, blend=0.7)
        self.assertEqual(source, "centroid")
        self.assertAlmostEqual(signal, 1.0, places=5)

    def test_interest_signal_blends_when_model_active(self):
        labeled, _ = self._add_labeled([1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        service = self._fake_service(labeled)
        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            train_learned_ranker(self.app)
        model = peek_learned_model()
        profile = InterestProfile(pos_centroid=_unit(0), neg_centroid=None, fingerprint=(1, 1, 1))

        blended, source = interest_signal(_unit(0), profile, model, blend=0.5)
        self.assertEqual(source, "learned")
        model_only, _ = interest_signal(_unit(0), None, model, blend=0.5)
        centroid_only = 1.0
        expected = 0.5 * model_only + 0.5 * centroid_only
        self.assertAlmostEqual(blended, expected, places=4)

    def test_model_status_reports_needed_ratings(self):
        status = model_status(self.app)
        self.assertFalse(status["available"])
        self.assertTrue(status["enabled"])
        self.assertEqual(status["needed_positive"], 5)


class RetrainHookTests(LearnedRankerTestCase):
    def _make_paper(self) -> Paper:
        paper = _paper("2606.99001")
        db.session.add(paper)
        db.session.commit()
        return paper

    def test_feedback_triggers_retrain_hook(self):
        paper = self._make_paper()
        from app.services.feedback import apply_feedback_action

        with patch("app.services.learned_ranker.train_learned_ranker") as mock_train:
            apply_feedback_action(paper.id, "save")
        # TESTING apps retrain synchronously inside request_retrain.
        mock_train.assert_called_once()

    def test_feedback_survives_training_failure(self):
        paper = self._make_paper()
        from app.services.feedback import apply_feedback_action

        with patch(
            "app.services.learned_ranker.train_learned_ranker",
            side_effect=RuntimeError("boom"),
        ):
            result = apply_feedback_action(paper.id, "save")

        self.assertTrue(result["active"])
        self.assertEqual(PaperFeedback.query.filter_by(paper_id=paper.id, action="save").count(), 1)

    def test_non_label_actions_do_not_retrain(self):
        paper = self._make_paper()
        from app.services.feedback import apply_feedback_action

        with patch("app.services.learned_ranker.request_retrain") as mock_request:
            apply_feedback_action(paper.id, "skimmed")
        mock_request.assert_not_called()

    def test_bulk_feedback_debounces_to_one_retrain(self):
        self.app.config["TESTING"] = False
        try:
            with patch("app.services.learned_ranker.train_learned_ranker") as mock_train:
                request_retrain(self.app)
                request_retrain(self.app)
                request_retrain(self.app)
                self.assertEqual(mock_train.call_count, 0)  # still pending
                self.assertTrue(flush_pending_retrain())
                self.assertEqual(mock_train.call_count, 1)
                self.assertFalse(flush_pending_retrain())  # nothing left
        finally:
            self.app.config["TESTING"] = True
            learned_ranker.cancel_pending_retrain()


def _entry(arxiv_id: str, title: str, embedding: np.ndarray | None = None, authors=None) -> dict:
    entry = {
        "arxiv_id": arxiv_id,
        "link": f"https://arxiv.org/abs/{arxiv_id}",
        "title": title,
        "author": ", ".join(authors or ["Some Author"]),
        "authors_list": authors or ["Some Author"],
        "abstract": "An abstract about things.",
        "api_affiliations": "",
        "pdf_affiliation_text": "",
        "publication_dt": date(2026, 6, 1),
    }
    if embedding is not None:
        entry["_embedding"] = embedding
    return entry


class DenseRetrievalCandidateTests(LearnedRankerTestCase):
    _WHITELISTS = {"authors": ["Jane Doe"], "titles": ["Diffusion"], "affiliations": ["MIT"]}

    def _generator(self, top_k=2, threshold=0.6, scorer=None, muted=None):
        settings = {
            "enabled": True,
            "blend": 0.7,
            "candidate_threshold": threshold,
            "candidate_top_k": top_k,
        }
        return WhitelistCandidateGenerator(
            whitelists=self._WHITELISTS,
            scraper_config={},
            muted=muted,
            interest_scorer=scorer or (lambda vec: float(vec[0])),
            interest_settings=settings,
        )

    def test_admits_above_threshold_with_interest_match_type(self):
        generator = self._generator()
        candidate = generator.process_single(_entry("2606.1", "Unrelated topic", _unit(0)))
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.match_types, ["Interest"])
        self.assertEqual(candidate.matched_terms, [])
        self.assertAlmostEqual(candidate.raw_features["interest_candidate_score"], 1.0, places=4)

    def test_rejects_below_threshold(self):
        generator = self._generator()
        low = np.full(DIM, 0.0, dtype=np.float32)
        low[0] = 0.3  # scorer returns 0.3 < 0.6
        self.assertIsNone(generator.process_single(_entry("2606.2", "Unrelated topic", low)))

    def test_respects_top_k_cap_per_generator(self):
        generator = self._generator(top_k=2)
        admitted = [
            generator.process_single(_entry(f"2606.3{i}", f"Interest paper {i}", _unit(0))) for i in range(4)
        ]
        self.assertEqual(sum(1 for c in admitted if c is not None), 2)

    def test_whitelist_match_still_wins_over_interest_gate(self):
        generator = self._generator()
        candidate = generator.process_single(
            _entry("2606.4", "A paper", _unit(0), authors=["Jane Doe"])
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.match_types, ["Author"])

    def test_muted_entries_are_not_admitted(self):
        muted = {"authors": ["Spam Author"], "affiliations": [], "topics": []}
        generator = self._generator(muted=muted)
        entry = _entry("2606.5", "Unrelated topic", _unit(0), authors=["Spam Author"])
        self.assertIsNone(generator.process_single(entry))

    def test_gate_inert_without_model_or_profile(self):
        # No injected scorer, no trained model, no cached profile → unchanged
        # behavior: non-matching entries are dropped.
        set_runtime_learned_prefs({"enabled": True, "blend": 0.7, "candidate_threshold": 0.6, "candidate_top_k": 10})
        generator = WhitelistCandidateGenerator(whitelists=self._WHITELISTS, scraper_config={})
        self.assertIsNone(generator.process_single(_entry("2606.6", "Unrelated topic", _unit(0))))

    def test_gate_uses_trained_model_from_module_state(self):
        labeled, _ = self._add_labeled([1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        service = self._fake_service(labeled)
        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            train_learned_ranker(self.app)
        set_runtime_learned_prefs({"enabled": True, "blend": 1.0, "candidate_threshold": 0.6, "candidate_top_k": 10})

        generator = WhitelistCandidateGenerator(whitelists=self._WHITELISTS, scraper_config={})
        on_topic = generator.process_single(_entry("2606.7", "Unrelated topic", _unit(0)))
        off_topic = generator.process_single(_entry("2606.8", "Unrelated topic", _unit(1)))
        self.assertIsNotNone(on_topic)
        self.assertEqual(on_topic.match_types, ["Interest"])
        self.assertIsNone(off_topic)

    def test_disabled_learned_ranking_disables_the_gate(self):
        settings = {"enabled": False, "blend": 0.7, "candidate_threshold": 0.6, "candidate_top_k": 10}
        generator = WhitelistCandidateGenerator(
            whitelists=self._WHITELISTS,
            scraper_config={},
            interest_settings=settings,
        )
        self.assertIsNone(generator.process_single(_entry("2606.9", "Unrelated topic", _unit(0))))


class ExplainHonestyTests(LearnedRankerTestCase):
    def _fake_model(self) -> LearnedModel:
        return LearnedModel(
            mean=np.zeros(DIM, dtype=np.float32),
            components=np.eye(2, DIM, dtype=np.float32),
            scale=np.ones(2, dtype=np.float32),
            coef=np.asarray([3.0, -3.0], dtype=np.float32),
            intercept=0.0,
            fingerprint=(1, 1, 1),
            trained_at="2026-07-01T00:00:00",
            n_positive=6,
            n_negative=4,
            n_weak=10,
        )

    def test_explain_score_labels_centroid_without_model(self):
        breakdown = explain_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 7, 1),
            resource_count=0,
            interest_similarity=0.8,
        )
        self.assertEqual(breakdown["interest_source"], "centroid")

    def test_explain_score_labels_learned_when_model_active(self):
        learned_ranker._set_cache(self._fake_model(), (1, 1, 1), "available")
        breakdown = explain_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 7, 1),
            resource_count=0,
            interest_similarity=0.8,
        )
        self.assertEqual(breakdown["interest_source"], "learned")

        factors = top_score_contributors(breakdown, limit=8)
        interest_factor = next(f for f in factors if f["key"] == "interest_bonus")
        self.assertEqual(interest_factor["label"], "Learned")

    def test_explain_score_has_no_source_without_interest_signal(self):
        breakdown = explain_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 7, 1),
            resource_count=0,
            interest_similarity=None,
        )
        self.assertIsNone(breakdown["interest_source"])

    def test_ranking_explanation_for_interest_match_type(self):
        paper = _paper("2606.7777")
        paper.match_type = "Interest"
        paper.matched_terms = []
        paper.interest_similarity = 0.8
        db.session.add(paper)
        db.session.commit()

        explanations = generate_ranking_explanation(paper, config=self.app.config["SCRAPER_CONFIG"])
        self.assertIn("Matched your learned interests", explanations)
        self.assertIn("Closely matches papers you saved", explanations)

        learned_ranker._set_cache(self._fake_model(), (1, 1, 1), "available")
        explanations = generate_ranking_explanation(paper, config=self.app.config["SCRAPER_CONFIG"])
        self.assertIn("Matches your learned interest model", explanations)

    def test_resolve_interest_source_respects_enabled_flag(self):
        learned_ranker._set_cache(self._fake_model(), (1, 1, 1), "available")
        self.assertEqual(resolve_interest_source(None), "learned")
        config = {"preferences": {"learned": {"enabled": False}}}
        self.assertEqual(resolve_interest_source(config), "centroid")


class SettingsRoundTripTests(LearnedRankerTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/settings")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def _post_preferences(self, extra: dict) -> object:
        data = {"csrf_token": self._csrf_token()}
        data.update(extra)
        return self.client.post("/settings/preferences", data=data)

    def test_settings_page_renders_learned_block(self):
        response = self.client.get("/settings")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="learned_enabled"', html)
        self.assertIn('id="learned-status"', html)
        self.assertIn('id="pref_learned_blend"', html)

    def test_toggle_and_blend_round_trip(self):
        response = self._post_preferences({"learned_enabled": "on", "pref_learned_blend": "0.4"})
        self.assertEqual(response.status_code, 302)
        learned = self.app.config["SCRAPER_CONFIG"]["preferences"]["learned"]
        self.assertTrue(learned["enabled"])
        self.assertAlmostEqual(learned["blend"], 0.4)

        response = self._post_preferences({"pref_learned_blend": "0.9"})  # checkbox off
        self.assertEqual(response.status_code, 302)
        learned = self.app.config["SCRAPER_CONFIG"]["preferences"]["learned"]
        self.assertFalse(learned["enabled"])
        self.assertAlmostEqual(learned["blend"], 0.9)

    def test_out_of_range_blend_is_rejected(self):
        response = self._post_preferences({"learned_enabled": "on", "pref_learned_blend": "1.5"})
        self.assertEqual(response.status_code, 302)  # redirect with flashed error
        learned = self.app.config["SCRAPER_CONFIG"]["preferences"].get("learned", {})
        self.assertNotEqual(learned.get("blend"), 1.5)


if __name__ == "__main__":
    unittest.main()
