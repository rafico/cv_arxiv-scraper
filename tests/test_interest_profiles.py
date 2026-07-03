"""Tests for multiple interest profiles (Wave 3).

Covers the migration-safety guarantee (a pre-Wave-3 DB gains a Default profile
that owns all existing NULL-profile feedback with zero data loss), the profile
CRUD/invariants, per-profile learned-model isolation, active switching, the
NL-description cold-start blend, per-profile digest sections, and the one-tap
token carrying + defaulting the profile id.
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
from sqlalchemy import inspect, text

from app.models import InterestProfile, Paper, PaperFeedback, db
from app.services import learned_ranker, profiles
from app.services.interest_model import reset_interest_profile_cache
from app.services.learned_ranker import (
    interest_signal,
    peek_learned_model,
    reset_learned_ranker_cache,
    train_learned_ranker,
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
    rng = np.random.default_rng(5000 + seed)
    vec = rng.normal(0, 1, DIM).astype(np.float32)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


class FakeEmbeddingService:
    def __init__(self, vectors_by_id: dict[int, np.ndarray]):
        self.vectors_by_id = vectors_by_id

    def index_size(self) -> int:
        return len(self.vectors_by_id)

    def encode(self, texts):
        # Deterministic-ish vector for a description string (only used for the
        # cold-start blend tests, which patch this with a fixed axis when needed).
        return np.asarray([_random_unit(hash(t) % 997) for t in texts], dtype=np.float32)

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


class ProfileTestBase(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        reset_learned_ranker_cache()
        reset_interest_profile_cache()
        self._faiss_dir = tempfile.mkdtemp(prefix="profiles_faiss_")
        self.addCleanup(shutil.rmtree, self._faiss_dir, True)
        self.app.config["FAISS_INDEX_DIR"] = self._faiss_dir

    def tearDown(self):
        reset_learned_ranker_cache()
        reset_interest_profile_cache()
        super().tearDown()


class MigrationSafetyTests(ProfileTestBase):
    """A pre-Wave-3 DB gains a Default profile that owns all existing feedback."""

    def _build_pre_wave3_schema(self) -> dict[int, np.ndarray]:
        db.drop_all()
        # Real papers table, but a pre-Wave-3 paper_feedback with NO profile_id and
        # NO interest_profiles table at all.
        Paper.__table__.create(bind=db.engine)
        db.session.execute(
            text(
                "CREATE TABLE paper_feedback ("
                "id INTEGER PRIMARY KEY, paper_id INTEGER NOT NULL, action VARCHAR(16) NOT NULL, "
                "reason VARCHAR(64), note TEXT, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
        )
        db.session.commit()

        vectors: dict[int, np.ndarray] = {}
        base = datetime(2026, 6, 1, 8, 0, 0)
        labels = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        for idx, label in enumerate(labels):
            paper = _paper(f"2606.{40000 + idx}")
            db.session.add(paper)
            db.session.flush()
            db.session.execute(
                text(
                    "INSERT INTO paper_feedback (paper_id, action, created_at) VALUES (:pid, :action, :ts)"
                ),
                {"pid": paper.id, "action": "save" if label else "skip", "ts": base + timedelta(hours=idx)},
            )
            vectors[paper.id] = _noisy_unit(0 if label else 1, seed=idx)
        db.session.commit()
        return vectors

    def test_ensure_schema_creates_default_owning_null_feedback(self):
        vectors = self._build_pre_wave3_schema()

        # Sanity: pre-migration there is no interest_profiles table and no profile_id.
        self.assertNotIn("interest_profiles", inspect(db.engine).get_table_names())

        from app.schema import ensure_schema

        ensure_schema()

        inspector = inspect(db.engine)
        self.assertIn("interest_profiles", inspector.get_table_names())
        feedback_cols = {c["name"] for c in inspector.get_columns("paper_feedback")}
        self.assertIn("profile_id", feedback_cols)

        # A single Default profile now exists, active + default.
        default = InterestProfile.query.filter_by(is_default=True).one()
        self.assertEqual(default.slug, "default")
        self.assertTrue(default.is_active)
        self.assertEqual(InterestProfile.query.count(), 1)

        # All pre-existing feedback is still present and still NULL (never rewritten),
        # and the default profile owns NULL rows.
        self.assertEqual(PaperFeedback.query.count(), 10)
        self.assertEqual(PaperFeedback.query.filter(PaperFeedback.profile_id.is_(None)).count(), 10)

        # The default profile trains from those NULL rows — nothing is lost.
        service = FakeEmbeddingService(dict(vectors))
        for idx in range(30):
            service.vectors_by_id[90000 + idx] = _random_unit(idx)
        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            status = train_learned_ranker(self.app)
        self.assertTrue(status["available"])
        self.assertEqual(status["n_positive"], 6)
        self.assertEqual(status["n_negative"], 4)

    def test_ensure_default_profile_is_idempotent(self):
        self._build_pre_wave3_schema()
        from app.schema import ensure_schema

        ensure_schema()
        ensure_schema()
        self.assertEqual(InterestProfile.query.filter_by(is_default=True).count(), 1)
        self.assertEqual(InterestProfile.query.filter_by(is_active=True).count(), 1)


class ProfileCrudTests(ProfileTestBase):
    def test_bootstrap_creates_single_default(self):
        default = profiles.ensure_default_profile()
        self.assertTrue(default.is_default)
        self.assertTrue(default.is_active)
        self.assertEqual(InterestProfile.query.count(), 1)

    def test_create_and_unique_slug(self):
        profiles.ensure_default_profile()
        a = profiles.create_profile("3D Reconstruction")
        b = profiles.create_profile("3D Reconstruction")
        self.assertEqual(a.slug, "3d-reconstruction")
        self.assertNotEqual(a.slug, b.slug)
        self.assertFalse(a.is_active)  # new profiles are inactive

    def test_set_active_is_exclusive(self):
        default = profiles.ensure_default_profile()
        other = profiles.create_profile("VLM efficiency")
        profiles.set_active_profile(other.id)
        self.assertEqual(InterestProfile.query.filter_by(is_active=True).count(), 1)
        self.assertEqual(profiles.get_active_profile().id, other.id)
        # default is no longer active
        self.assertFalse(db.session.get(InterestProfile, default.id).is_active)

    def test_cannot_delete_default(self):
        default = profiles.ensure_default_profile()
        with self.assertRaises(ValueError):
            profiles.delete_profile(default.id)

    def test_cannot_delete_last_profile(self):
        profiles.ensure_default_profile()
        # Only the default exists; deleting it is doubly forbidden.
        only = InterestProfile.query.one()
        with self.assertRaises(ValueError):
            profiles.delete_profile(only.id)

    def test_delete_rehomes_feedback_to_default(self):
        default = profiles.ensure_default_profile()
        other = profiles.create_profile("Temp")
        paper = _paper("2606.50001")
        db.session.add(paper)
        db.session.commit()
        db.session.add(PaperFeedback(paper_id=paper.id, action="save", profile_id=other.id))
        db.session.commit()

        profiles.delete_profile(other.id)
        self.assertIsNone(db.session.get(InterestProfile, other.id))
        # The feedback survived and now belongs to the default profile.
        row = PaperFeedback.query.filter_by(paper_id=paper.id, action="save").one()
        self.assertEqual(row.profile_id, default.id)

    def test_delete_active_falls_back_to_default(self):
        default = profiles.ensure_default_profile()
        other = profiles.create_profile("Temp")
        profiles.set_active_profile(other.id)
        profiles.delete_profile(other.id)
        self.assertEqual(profiles.get_active_profile().id, default.id)


class PerProfileIsolationTests(ProfileTestBase):
    def _seed_profile(self, ref, axis_pos, axis_neg, base_id):
        vectors: dict[int, np.ndarray] = {}
        base = datetime(2026, 6, 1, 8, 0, 0)
        labels = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        for idx, label in enumerate(labels):
            paper = _paper(f"26{base_id}.{idx:05d}")
            db.session.add(paper)
            db.session.flush()
            db.session.add(
                PaperFeedback(
                    paper_id=paper.id,
                    action="save" if label else "skip",
                    profile_id=ref.id,
                    created_at=base + timedelta(hours=idx),
                )
            )
            vectors[paper.id] = _noisy_unit(axis_pos if label else axis_neg, seed=base_id * 100 + idx)
        db.session.commit()
        return vectors

    def test_profiles_train_on_their_own_feedback(self):
        default = profiles.ensure_default_profile()
        other = profiles.create_profile("Other")
        default_ref = profiles.profile_ref(default)
        other_ref = profiles.profile_ref(other)

        # Profile A likes axis 0, profile B likes axis 2 — disjoint feedback.
        va = self._seed_profile(default_ref, 0, 1, base_id=10)
        vb = self._seed_profile(other_ref, 2, 3, base_id=20)
        all_vectors = {**va, **vb}
        for idx in range(30):
            all_vectors[999000 + idx] = _random_unit(idx)
        service = FakeEmbeddingService(all_vectors)

        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            train_learned_ranker(self.app, profile=default_ref)
            train_learned_ranker(self.app, profile=other_ref)

        model_a = peek_learned_model(profile=default_ref)
        model_b = peek_learned_model(profile=other_ref)
        self.assertIsNotNone(model_a)
        self.assertIsNotNone(model_b)

        # Profile A scores its liked axis (0) higher than B's liked axis (2), and
        # vice-versa: the two models learned different tastes from isolated data.
        a_on_0 = float(model_a.predict_proba(np.asarray([_unit(0)]))[0])
        a_on_2 = float(model_a.predict_proba(np.asarray([_unit(2)]))[0])
        b_on_0 = float(model_b.predict_proba(np.asarray([_unit(0)]))[0])
        b_on_2 = float(model_b.predict_proba(np.asarray([_unit(2)]))[0])
        self.assertGreater(a_on_0, a_on_2)
        self.assertGreater(b_on_2, b_on_0)

    def test_default_and_nondefault_use_distinct_artifacts(self):
        default = profiles.ensure_default_profile()
        other = profiles.create_profile("Other")
        default_path = learned_ranker._artifact_path(profiles.profile_ref(default))
        other_path = learned_ranker._artifact_path(profiles.profile_ref(other))
        self.assertEqual(default_path.name, "learned_ranker.npz")
        self.assertEqual(other_path.name, f"learned_ranker_{other.slug}.npz")


class DescriptionColdStartTests(ProfileTestBase):
    def test_description_only_profile_ranks_by_cosine(self):
        # No model, no centroid: a description vector alone drives the signal.
        description_vector = _unit(7)
        near, source = interest_signal(_unit(7), None, None, blend=0.7, description_vector=description_vector)
        far, _ = interest_signal(_unit(9), None, None, blend=0.7, description_vector=description_vector)
        self.assertEqual(source, "description")
        self.assertAlmostEqual(near, 1.0, places=5)
        self.assertGreater(near, far)

    def test_description_blends_with_model_signal(self):
        # With a model present the description is averaged in (source stays honest).
        model_only, _ = interest_signal(_unit(0), None, _FakeModel(), blend=0.7)
        blended, source = interest_signal(_unit(0), None, _FakeModel(), blend=0.7, description_vector=_unit(0))
        self.assertEqual(source, "learned")
        # description cosine on _unit(0) is 1.0, averaged with the model signal.
        self.assertAlmostEqual(blended, 0.5 * model_only + 0.5 * 1.0, places=5)


class _FakeModel:
    """Minimal stand-in returning a fixed probability so signal math is checkable."""

    def predict_proba(self, vectors):
        return np.full((np.asarray(vectors).reshape(-1, DIM).shape[0],), 0.8, dtype=np.float32)


class OneTapTokenProfileTests(ProfileTestBase):
    def test_token_round_trips_profile_id(self):
        from app.services.email_digest import load_one_tap_token, make_one_tap_token

        token = make_one_tap_token(self.app, 5, "save", 7, profile_id=42)
        data = load_one_tap_token(self.app, token)
        self.assertEqual(data["p"], 5)
        self.assertEqual(data["pr"], 42)

    def test_token_without_profile_id_is_backward_compatible(self):
        from app.services.email_digest import load_one_tap_token, make_one_tap_token

        token = make_one_tap_token(self.app, 5, "save")
        data = load_one_tap_token(self.app, token)
        self.assertNotIn("pr", data)  # caller defaults it to the default profile

    def test_one_tap_route_attributes_to_default_when_missing(self):
        profiles.ensure_default_profile()
        default = profiles.get_default_profile()
        paper = _paper("2606.60001")
        db.session.add(paper)
        db.session.commit()

        from app.services.email_digest import make_one_tap_token

        token = make_one_tap_token(self.app, paper.id, "save")  # no profile id
        client = self.app.test_client()
        resp = client.get(f"/api/feedback/one-tap?token={token}")
        self.assertEqual(resp.status_code, 200)
        row = PaperFeedback.query.filter_by(paper_id=paper.id, action="save").one()
        self.assertEqual(row.profile_id, default.id)


class DigestSectionTests(ProfileTestBase):
    def test_multiple_profiles_produce_sections_with_profile_tokens(self):
        from app.services.email_digest import build_digest_preview

        default = profiles.ensure_default_profile()
        other = profiles.create_profile("VLM efficiency")

        # Two recent papers so both sections have content.
        from app.services.text import now_utc

        papers = []
        for i in range(2):
            paper = _paper(f"2606.7000{i}")
            paper.scraped_at = now_utc()
            paper.paper_score = 5.0
            db.session.add(paper)
            papers.append(paper)
        db.session.commit()

        preview = build_digest_preview(self.app)
        sections = preview["sections"]
        self.assertIsNotNone(sections)
        names = {s["name"] for s in sections}
        self.assertIn(default.name, names)
        self.assertIn(other.name, names)
        # Each section carries its owning profile id (for one-tap attribution).
        for section in sections:
            self.assertIn(section["profile_id"], {default.id, other.id})
        # The rendered HTML carries a section header labeled with a profile name.
        self.assertIn(other.name, preview["html"])

    def test_single_profile_keeps_flat_layout(self):
        from app.services.email_digest import build_digest_preview

        profiles.ensure_default_profile()
        preview = build_digest_preview(self.app)
        self.assertIsNone(preview["sections"])  # unchanged single-list behavior


class ProfileApiTests(ProfileTestBase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_list_create_activate_delete_flow(self):
        token = self._csrf_token()
        # List includes the auto-created default.
        resp = self.client.get("/api/profiles")
        self.assertEqual(resp.status_code, 200)
        self.assertGreaterEqual(len(resp.get_json()["profiles"]), 1)

        # Create.
        resp = self.client.post(
            "/api/profiles", json={"name": "3D Recon", "description": "gaussian splatting"},
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(resp.status_code, 201)
        new_id = resp.get_json()["id"]

        # Activate.
        resp = self.client.post(f"/api/profiles/{new_id}/activate", headers={"X-CSRF-Token": token})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()["is_active"])
        self.assertEqual(profiles.get_active_profile().id, new_id)

        # Delete (guarded ok for a non-default).
        resp = self.client.delete(f"/api/profiles/{new_id}", headers={"X-CSRF-Token": token})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()["deleted"])

    def test_delete_default_rejected_by_api(self):
        token = self._csrf_token()
        default_id = profiles.get_default_profile().id
        resp = self.client.delete(f"/api/profiles/{default_id}", headers={"X-CSRF-Token": token})
        self.assertEqual(resp.status_code, 400)

    def test_create_requires_csrf(self):
        resp = self.client.post("/api/profiles", json={"name": "X"})
        self.assertIn(resp.status_code, (400, 403))
