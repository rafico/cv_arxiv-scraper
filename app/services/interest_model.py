"""Interest profile in SPECTER2 embedding space: collections first, feedback as fallback.

Collections that are large enough are the interest model: one mean-centred
centroid per collection, and a paper's affinity is its best z-score against them
(:func:`fit_collection_profile`, pure NumPy; measured by
``scripts/eval_collection_affinity.py``). Without them the profile comes from
feedback: positive/negative centroids from save/priority vs skip/ignore, scored
by cosine similarity. Cold-starts inert: with no qualifying collection and fewer
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
MIN_POSITIVE_FEEDBACK = 5  # also the fewest members a collection needs for a centroid
MIN_NEGATIVE_FEEDBACK = 3

# Collection affinity. z = how many standard deviations closer a paper is to a
# collection than papers in no collection are.
AFFINITY_Z_SCALE = 4.0  # stored signal = clip(z / scale, -1, 1), so 0.5 means z = 2
# ponytail: one admission floor for all collections, so recall is uneven (a broad
# collection keeps far fewer of its own papers than a narrow one). Upgrade:
# per-collection thresholds, tuned with the eval script's holdout table.
AFFINITY_Z_MIN = 2.0
MIN_BACKGROUND = 200  # papers outside every collection needed for the z statistics

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
    """Interest centroids in embedding space (L2-normalized).

    A feedback profile sets ``pos_centroid``; a collection profile sets ``centroids``
    and the fields after it, one row per collection in ``collection_ids`` order.
    """

    pos_centroid: np.ndarray | None
    neg_centroid: np.ndarray | None
    fingerprint: tuple[int, ...]
    mean: np.ndarray | None = None  # corpus mean every vector is centred by
    centroids: np.ndarray | None = None  # (k, dim) centroids of the centred member vectors
    bg_mean: np.ndarray | None = None  # (k,) mean centred cosine of the background
    bg_std: np.ndarray | None = None  # (k,) its standard deviation, floored
    collection_ids: tuple[int, ...] = ()
    labels: tuple[str, ...] = ()  # collection names, as shown in "why this paper"


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


def _membership_fingerprint() -> tuple[int, ...]:
    """Cheap cache key over the collections: membership counts and id sums, and the names.

    ponytail: sums, not the rows, so a set of edits between two builds that keeps every
    count and sum goes unseen until the index grows. Upgrade: hash the
    (collection_id, paper_id, decision) rows.
    """
    from app.models import Collection, PaperCollection, db, in_review_clause

    in_review = in_review_clause()
    rows = (
        db.session.query(
            # In review and excluded apart: an excluded row moves no centroid, but it
            # takes its paper out of the background.
            in_review,
            db.func.count(PaperCollection.id),
            db.func.sum(PaperCollection.paper_id),
            db.func.sum(PaperCollection.collection_id),
            # Two papers swapping collections keep both sums; this one moves.
            db.func.sum(PaperCollection.paper_id * PaperCollection.collection_id),
        )
        .group_by(in_review)
        .order_by(in_review)
        .all()
    )
    # A rename counts: the gate stores the collection's name on the papers it admits.
    # hash() differs from process to process, which is all a per-process cache needs.
    names = db.session.query(Collection.id, Collection.name).order_by(Collection.id).all()
    return (*(int(value) for row in rows for value in row), hash(tuple(map(tuple, names))))


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


def _centred_unit(vectors, mean: np.ndarray) -> np.ndarray:
    """Each vector minus the corpus mean, L2-normalized: (n, dim)."""
    import numpy as np

    centred = np.atleast_2d(np.asarray(vectors, dtype=np.float32)) - mean
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    return centred / np.where(norms == 0.0, 1.0, norms)


def _centred_cosine(vectors, mean: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Cosine of each mean-centred vector to each centroid: (n, k)."""
    return _centred_unit(vectors, mean) @ centroids.T


def fit_collection_profile(
    members: dict[int, np.ndarray],
    background: np.ndarray,
    mean: np.ndarray,
    *,
    labels: dict[int, str] | None = None,
    fingerprint: tuple[int, ...] = (),
) -> InterestProfile | None:
    """Fit the collection-affinity model from plain arrays (no DB, no embedding service).

    ``members`` maps a collection id to its (n, dim) member vectors, ``background``
    holds the vectors of papers in no collection, and ``mean`` is the corpus mean
    every vector is centred by. A collection with at least MIN_POSITIVE_FEEDBACK
    members gets one row: the normalized mean of its centred members, plus the mean
    and standard deviation of the background's centred cosine to it. None when the
    background is under MIN_BACKGROUND or no collection qualifies.
    """
    import numpy as np

    background = np.asarray(background, dtype=np.float32)
    if len(background) < MIN_BACKGROUND:
        return None
    mean = np.asarray(mean, dtype=np.float32)
    rows: dict[int, np.ndarray] = {}
    for collection_id, vectors in sorted(members.items()):
        if len(vectors) >= MIN_POSITIVE_FEEDBACK:
            centroid = _normalized_centroid(np.asarray(vectors, dtype=np.float32) - mean)
            if centroid is not None:
                rows[collection_id] = centroid
    if not rows:
        return None
    centroids = np.vstack(list(rows.values()))
    raw = _centred_cosine(background, mean, centroids)
    return InterestProfile(
        pos_centroid=None,
        neg_centroid=None,
        fingerprint=fingerprint,
        mean=mean,
        centroids=centroids,
        bg_mean=raw.mean(axis=0),
        bg_std=np.maximum(raw.std(axis=0), 1e-6),
        collection_ids=tuple(rows),
        labels=tuple(str((labels or {}).get(collection_id, collection_id)) for collection_id in rows),
    )


def affinity_scores(model: InterestProfile, vectors) -> np.ndarray:
    """z of each vector against each collection: (n, k), columns in ``model.collection_ids`` order.

    ``vectors`` must be L2-normalized, as the index stores them: the mean is subtracted
    as is, so another length gives another z.
    """
    return (_centred_cosine(vectors, model.mean, model.centroids) - model.bg_mean) / model.bg_std


def collection_affinity(model: InterestProfile, vector) -> tuple[float, int]:
    """A paper's affinity: its best z over the collections, and that collection's id."""
    import numpy as np

    z = affinity_scores(model, vector)[0]
    best = int(np.argmax(z))
    return float(z[best]), model.collection_ids[best]


def nearest_member(model: InterestProfile, vectors, members) -> np.ndarray:
    """For each vector the row of ``members`` closest to it: (n,).

    Cosine in the scorer's space, both sides centred by the corpus mean. A member
    row that is not finite never wins.
    """
    import numpy as np

    cosine = _centred_unit(vectors, model.mean) @ _centred_unit(members, model.mean).T
    return np.nan_to_num(cosine, nan=-np.inf).argmax(axis=1)


def _collection_profile(service, fingerprint: tuple[int, ...]) -> InterestProfile | None:
    """Collection profile from the in-review memberships; None falls through to feedback."""
    import numpy as np

    from app.enums import MatchType
    from app.models import Collection, Paper, PaperCollection, db, in_review_clause

    # ponytail: every in-review membership counts at once, whoever added it (a query
    # import, an agent, the owner). Upgrade: a created-by marker and centroids from
    # owner-confirmed rows only, if the eval's checkpoint shows the centroids drifting.
    members: dict[int, list[int]] = {}
    for collection_id, paper_id in (
        db.session.query(PaperCollection.collection_id, PaperCollection.paper_id)
        .join(Paper, Paper.id == PaperCollection.paper_id)
        .filter(in_review_clause())
    ):
        members.setdefault(collection_id, []).append(paper_id)
    if not any(len(paper_ids) >= MIN_POSITIVE_FEEDBACK for paper_ids in members.values()):
        return None

    # ponytail: reads every indexed vector on each rebuild (once per fingerprint change;
    # a tenth of a second at a few thousand papers). Upgrade: a sampled background.
    papers = db.session.query(Paper.id, Paper.match_type).all()
    found_ids, vectors = service.get_paper_vectors([paper_id for paper_id, _ in papers])
    # The corpus mean runs over every row: a single non-finite one would turn each z into NaN.
    finite = np.isfinite(vectors).all(axis=1)
    found_ids, vectors = [paper_id for paper_id, ok in zip(found_ids, finite) if ok], vectors[finite]
    if not found_ids:
        return None
    row = {paper_id: index for index, paper_id in enumerate(found_ids)}
    # Background: papers with no membership row at all (an excluded paper is not
    # "unrelated") and not admitted by this gate itself, or its own admissions
    # would drift into the statistics it is measured against.
    in_a_collection = {paper_id for (paper_id,) in db.session.query(PaperCollection.paper_id)}
    background = [
        row[paper_id]
        for paper_id, match_type in papers
        if paper_id in row and paper_id not in in_a_collection and match_type != MatchType.INTEREST.value
    ]
    return fit_collection_profile(
        {
            collection_id: vectors[[row[paper_id] for paper_id in paper_ids if paper_id in row]]
            for collection_id, paper_ids in members.items()
        },
        vectors[background],
        vectors.mean(axis=0),
        labels=dict(db.session.query(Collection.id, Collection.name)),
        fingerprint=fingerprint,
    )


def build_interest_profile(app, profile=None) -> InterestProfile | None:
    """Build (or return cached) the interest profile: collections, else a profile's feedback.

    ``profile`` is a :class:`app.services.profiles.ProfileRef`; when None the
    active profile is resolved so the centroid tracks whichever profile the feed
    is currently showing. Returns None until a collection qualifies or enough
    positive feedback exists — callers treat that as "feature disabled". Never
    raises: degrades to None.
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
            # would stick. Keying on index size too forces the recompute. The
            # membership key does the same for the collection profile.
            fingerprint = (*_feedback_fingerprint(ref), service.index_size(), *_membership_fingerprint())
            with _cache_lock:
                if _cached_fingerprints.get(key) == fingerprint and key in _cached_profiles:
                    _cached_active_key = key
                    return _cached_profiles[key]

            # ponytail: collections are global, so every interest profile gets the same
            # collection rows. Upgrade: per-profile collection sets, if profiles diverge.
            profile_obj = _collection_profile(service, fingerprint)
            if profile_obj is None:
                pos_ids = _paper_ids_for_actions(POSITIVE_ACTIONS, ref)
                _, pos_vectors = service.get_paper_vectors(pos_ids)
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
    """Interest similarity in [-1, 1] for an L2-normalized vector.

    Cosine-based for a feedback profile; for a collection profile the best
    collection z over AFFINITY_Z_SCALE, so 0.5 means z = 2.
    """
    import numpy as np

    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return 0.0
    vec = vec / norm

    if profile.centroids is not None:
        similarity = collection_affinity(profile, vec)[0] / AFFINITY_Z_SCALE
    else:
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
    load). A collection profile scores alone; a feedback profile blends in the
    learned ranker's probability when that model is active (see
    app/services/learned_ranker.py). Clears similarities when no signal source
    exists anymore.

    ponytail: nothing calls this when a collection changes, so stored
    similarities follow collection edits only at the next ``cv-arxiv-backfill
    interest`` or Settings save (new papers are scored fresh at scrape time).
    Upgrade: rescore after collection writes.
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
        # A collection profile scores alone (see interest_signal); embedding the
        # description for it would only load the encoder into this process.
        if profile is None or profile.centroids is None:
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
