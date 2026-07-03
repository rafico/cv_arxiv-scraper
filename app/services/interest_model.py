"""Learned interest profile from feedback + SPECTER2 embeddings.

Builds positive/negative interest centroids from save/priority vs skip/ignore
feedback and scores papers by cosine similarity. Cold-starts inert: with fewer
than MIN_POSITIVE_FEEDBACK indexed positive papers no profile exists and the
ranking feature contributes exactly 0.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

LOGGER = logging.getLogger(__name__)

POSITIVE_ACTIONS = ("save", "priority")
NEGATIVE_ACTIONS = ("skip", "ignore")
MIN_POSITIVE_FEEDBACK = 5
MIN_NEGATIVE_FEEDBACK = 3

# Per-interest-profile centroid caches, keyed by profile id (None-key = the
# "no DB profile" fallback that behaves like the old single global model).
_cache_lock = threading.Lock()
_cached_profiles: dict[object, InterestProfile | None] = {}
_cached_fingerprints: dict[object, tuple[int, ...]] = {}
# Profile id whose centroid was most recently built for the active feed, so
# scrape worker threads (no app context) can read it via get_cached_interest_profile.
_cached_active_key: object = None


@dataclass(slots=True)
class InterestProfile:
    """Interest centroids in embedding space (L2-normalized)."""

    pos_centroid: np.ndarray
    neg_centroid: np.ndarray | None
    fingerprint: tuple[int, ...]


def _profile_condition(ref):
    """Predicate for feedback rows owned by ``ref`` (default owns NULL rows)."""
    from app.models import PaperFeedback, db

    if ref is None or getattr(ref, "id", None) is None:
        return None
    conditions = [PaperFeedback.profile_id == ref.id]
    if getattr(ref, "is_default", False):
        conditions.append(PaperFeedback.profile_id.is_(None))
    return db.or_(*conditions)


def _feedback_fingerprint(ref=None) -> tuple[int, int]:
    """Cheap cache key over relevant feedback rows: (count, max id)."""
    from app.models import PaperFeedback, db

    query = db.session.query(db.func.count(PaperFeedback.id), db.func.max(PaperFeedback.id)).filter(
        PaperFeedback.action.in_(POSITIVE_ACTIONS + NEGATIVE_ACTIONS)
    )
    condition = _profile_condition(ref)
    if condition is not None:
        query = query.filter(condition)
    count, max_id = query.one()
    return int(count or 0), int(max_id or 0)


def _paper_ids_for_actions(actions: tuple[str, ...], ref=None) -> list[int]:
    from app.models import PaperFeedback, db

    query = db.session.query(PaperFeedback.paper_id).filter(PaperFeedback.action.in_(actions))
    condition = _profile_condition(ref)
    if condition is not None:
        query = query.filter(condition)
    return [row[0] for row in query.distinct().all()]


def _normalized_centroid(vectors: np.ndarray) -> np.ndarray | None:
    import numpy as np

    if vectors.shape[0] == 0:
        return None
    centroid = vectors.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        return None
    return (centroid / norm).astype(np.float32)


def build_interest_profile(app, profile=None) -> InterestProfile | None:
    """Build (or return cached) interest centroids from a profile's feedback.

    ``profile`` is a :class:`app.services.profiles.ProfileRef`; when None the
    active profile is resolved so the centroid tracks whichever profile the feed
    is currently showing. Returns None until enough positive feedback exists —
    callers treat that as "feature disabled". Never raises: degrades to None.
    """
    global _cached_active_key

    try:
        with app.app_context():
            ref = profile
            if ref is None:
                try:
                    # Resolve the active profile AND publish it to the learned-ranker
                    # runtime snapshot, so scrape worker threads (no app context)
                    # score candidates against the active profile's model/description.
                    from app.services import learned_ranker

                    ref = learned_ranker._resolve_active_ref(app)
                except Exception:  # pragma: no cover - degrade to the global model
                    ref = None
            key = getattr(ref, "id", None)

            from app.services.embeddings import get_embedding_service

            service = get_embedding_service(app)
            # Fold the index size into the cache key: a paper saved before it was
            # embedded yields no vector, so a profile can read as "disabled"
            # (None) at 5 saves. Once the backlog embeds (index grows) with no new
            # feedback, the feedback-only key wouldn't change and the stale None
            # would stick. Keying on index size too forces the recompute.
            fingerprint = (*_feedback_fingerprint(ref), service.index_size())
            with _cache_lock:
                if _cached_fingerprints.get(key) == fingerprint and key in _cached_profiles:
                    _cached_active_key = key
                    return _cached_profiles[key]

            pos_ids = _paper_ids_for_actions(POSITIVE_ACTIONS, ref)
            _, pos_vectors = service.get_paper_vectors(pos_ids)
            profile_obj: InterestProfile | None = None
            if pos_vectors.shape[0] >= MIN_POSITIVE_FEEDBACK:
                pos_centroid = _normalized_centroid(pos_vectors)
                if pos_centroid is not None:
                    neg_ids = _paper_ids_for_actions(NEGATIVE_ACTIONS, ref)
                    _, neg_vectors = service.get_paper_vectors(neg_ids)
                    neg_centroid = (
                        _normalized_centroid(neg_vectors) if neg_vectors.shape[0] >= MIN_NEGATIVE_FEEDBACK else None
                    )
                    profile_obj = InterestProfile(
                        pos_centroid=pos_centroid,
                        neg_centroid=neg_centroid,
                        fingerprint=fingerprint,
                    )

            with _cache_lock:
                _cached_profiles[key] = profile_obj
                _cached_fingerprints[key] = fingerprint
                _cached_active_key = key
            return profile_obj
    except Exception:
        LOGGER.warning("Interest profile build failed (non-fatal)", exc_info=True)
        return None


def score_vector(profile: InterestProfile, vector: np.ndarray) -> float:
    """Cosine-based interest similarity in [-1, 1] for an L2-normalized vector."""
    import numpy as np

    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return 0.0
    vec = vec / norm

    similarity = float(np.dot(vec, profile.pos_centroid))
    if profile.neg_centroid is not None:
        similarity -= float(np.dot(vec, profile.neg_centroid))
    return max(-1.0, min(1.0, similarity))


def get_cached_interest_profile() -> InterestProfile | None:
    """Last profile built by :func:`build_interest_profile`, without any DB access.

    Scrape worker threads (no app context) read this after the scrape engine
    warms the cache at scrape start; returns None when nothing is cached.
    """
    with _cache_lock:
        return _cached_profiles.get(_cached_active_key)


def reset_interest_profile_cache() -> None:
    """Reset the module cache (for testing)."""
    global _cached_active_key
    with _cache_lock:
        _cached_profiles.clear()
        _cached_fingerprints.clear()
        _cached_active_key = None


def recompute_interest_similarities(app, *, batch_size: int = 500) -> int:
    """Refresh Paper.interest_similarity for all indexed papers, then rescore.

    Cheap when a profile exists (vectors come from FAISS reconstruct, no model
    load). Blends in the learned ranker's probability when that model is
    active (see app/services/learned_ranker.py); clears similarities when
    neither signal source exists anymore.
    """
    profile = build_interest_profile(app)

    from app.models import Paper, db

    # Learned-model blend (best-effort: any failure degrades to centroid-only).
    learned_model = None
    description_vector = None
    blend = 0.7
    try:
        from app.services.learned_ranker import (
            active_description_vector,
            ensure_learned_model,
            learned_preferences,
        )

        with app.app_context():
            learned_prefs = learned_preferences(app.config.get("SCRAPER_CONFIG"))
        blend = float(learned_prefs.get("blend", 0.7))
        if learned_prefs.get("enabled", True):
            learned_model = ensure_learned_model(app)
        description_vector = active_description_vector(app)
    except Exception:
        LOGGER.warning("Learned-ranker lookup failed (non-fatal); using centroid only", exc_info=True)

    service = None
    if profile is not None or learned_model is not None or description_vector is not None:
        from app.services.embeddings import get_embedding_service

        service = get_embedding_service(app)

    updated = 0
    with app.app_context():
        offset = 0
        while True:
            papers = Paper.query.order_by(Paper.id).offset(offset).limit(batch_size).all()
            if not papers:
                break
            if service is None:
                for paper in papers:
                    if paper.interest_similarity is not None:
                        paper.interest_similarity = None
                        updated += 1
            else:
                from app.services.learned_ranker import interest_signal

                found_ids, vectors = service.get_paper_vectors([paper.id for paper in papers])
                similarity_by_id = {}
                for idx, paper_id in enumerate(found_ids):
                    signal, _source = interest_signal(vectors[idx], profile, learned_model, blend, description_vector)
                    if signal is not None:
                        similarity_by_id[paper_id] = signal
                for paper in papers:
                    similarity = similarity_by_id.get(paper.id)
                    if similarity is not None:
                        paper.interest_similarity = round(similarity, 4)
                        updated += 1
            db.session.commit()
            offset += batch_size

    from app.services.ranking import recompute_all_paper_scores

    recompute_all_paper_scores(app)
    return updated
