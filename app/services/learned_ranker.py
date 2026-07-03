"""Learned per-user ranker: logistic regression over PCA-compressed embeddings.

Implements the Scholar Inbox production recipe (arXiv:2504.08385, ACL 2025):

* positives  = papers with save / priority / shared feedback,
* explicit negatives = skip / ignore feedback,
* weak negatives     = a few thousand random unlabeled corpus papers with a
  down-scaled weight (``RANDOM_NEGATIVE_RATIO`` of the positive mass divided by
  ``NEGATIVE_WEIGHT_SCALE``),
* features   = SPECTER2 embeddings PCA-compressed to ~256 dims and scaled,
* model      = L2-regularized logistic regression (C≈0.12), retrained from
  scratch on every feedback event (milliseconds on CPU at this scale),
* temporal decay: old ratings are down-weighted with a one-year half-life.

Cold start: below ``MIN_POSITIVE_FEEDBACK`` embedded positives the model
reports unavailable and callers fall back to the centroid interest profile
(:mod:`app.services.interest_model`), so behavior is unchanged for new users.

Artifacts (PCA mean/components/scale + LR weights) persist as a single ``.npz``
next to the FAISS index, written under the same cross-process file lock the
index uses, so scoring works without an app context (scrape worker threads)
and across restarts. Everything degrades gracefully: no failure in here may
break feedback handling, a scrape, or a page render.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from app.services.interest_model import MIN_POSITIVE_FEEDBACK

if TYPE_CHECKING:
    import numpy as np

    from app.services.interest_model import InterestProfile

LOGGER = logging.getLogger(__name__)

# Recipe labels (positives include "shared", unlike the centroid profile).
POSITIVE_ACTIONS = ("save", "priority", "shared")
NEGATIVE_ACTIONS = ("skip", "ignore")

# Scholar Inbox hyperparameters (see module docstring).
PCA_COMPONENTS = 256
LR_C = 0.12
RANDOM_NEGATIVE_RATIO = 0.8
NEGATIVE_WEIGHT_SCALE = 5.0
MAX_WEAK_NEGATIVES = 3000
RATING_HALF_LIFE_DAYS = 365.0

ARTIFACT_FILENAME = "learned_ranker.npz"
_ARTIFACT_VERSION = 1

# Trailing debounce for the retrain-on-feedback hook (seconds).
RETRAIN_DEBOUNCE_SECONDS = 2.0

_cache_lock = threading.Lock()
_cached_model: LearnedModel | None = None
_cached_fingerprint: tuple[int, int, int] | None = None
# "unknown" (never trained this process) | "available" | "unavailable"
_cache_state = "unknown"

# Disk-artifact read cache: (path, mtime_ns, size) -> model, so request-path
# peeks cost one stat() instead of a parse.
_artifact_cache_key: tuple[str, int, int] | None = None
_artifact_cache_model: LearnedModel | None = None

# Snapshot of preferences["learned"] taken whenever a caller that HAS the
# product config passes through (feature extractor construction, training,
# status). The candidate-generation gate runs in scrape worker threads with no
# app context and reads this snapshot instead.
_runtime_prefs_lock = threading.Lock()
_runtime_prefs: dict | None = None

_retrain_lock = threading.Lock()
_pending_retrain: threading.Timer | None = None


@dataclass(slots=True)
class LearnedModel:
    """PCA projection + logistic-regression weights, all numpy arrays."""

    mean: np.ndarray  # (768,)
    components: np.ndarray  # (k, 768)
    scale: np.ndarray  # (k,)
    coef: np.ndarray  # (k,)
    intercept: float
    fingerprint: tuple[int, int, int]
    trained_at: str
    n_positive: int
    n_negative: int
    n_weak: int

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        import numpy as np

        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        return ((vecs - self.mean) @ self.components.T) / self.scale

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        import numpy as np

        z = self.transform(vectors) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


# ── preferences plumbing ────────────────────────────────────────────────


def default_learned_preferences() -> dict:
    from app.services.preferences import DEFAULT_PREFERENCES

    return dict(DEFAULT_PREFERENCES["learned"])


def learned_preferences(config: dict | None) -> dict:
    """Resolve the ``preferences.learned`` block (and refresh the snapshot)."""
    from app.services.preferences import get_preferences

    prefs = get_preferences(config)["learned"]
    if isinstance(config, dict):
        set_runtime_learned_prefs(prefs)
    return prefs


def set_runtime_learned_prefs(prefs: dict) -> None:
    global _runtime_prefs
    with _runtime_prefs_lock:
        _runtime_prefs = dict(prefs)


def get_runtime_learned_prefs() -> dict:
    """Last-seen learned preferences (defaults until any config passes by)."""
    with _runtime_prefs_lock:
        if _runtime_prefs is not None:
            return dict(_runtime_prefs)
    return default_learned_preferences()


# ── artifact persistence ────────────────────────────────────────────────


def _artifact_path() -> Path | None:
    """Resolve the artifact location without ever creating heavy services.

    Prefers the active app's FAISS dir, then the already-created embedding
    service singleton (scrape worker threads have no app context), then the
    FAISS_INDEX_DIR env var. Returns None when nothing is resolvable — the
    learned ranker then simply reports unavailable.
    """
    try:
        from flask import current_app, has_app_context

        if has_app_context():
            index_dir = current_app.config.get(
                "FAISS_INDEX_DIR",
                str(Path(current_app.instance_path) / "faiss_index"),
            )
            return Path(index_dir) / ARTIFACT_FILENAME
    except Exception:  # pragma: no cover - flask always importable in-app
        pass

    from app.services.embeddings import peek_embedding_service

    service = peek_embedding_service()
    if service is not None:
        return Path(service.index_dir) / ARTIFACT_FILENAME

    env_dir = os.environ.get("FAISS_INDEX_DIR", "").strip()
    if env_dir:
        return Path(env_dir) / ARTIFACT_FILENAME
    return None


def _save_artifact(model: LearnedModel, path: Path) -> None:
    import numpy as np

    from app.services.embeddings import _index_file_lock

    meta = {
        "version": _ARTIFACT_VERSION,
        "fingerprint": list(model.fingerprint),
        "trained_at": model.trained_at,
        "n_positive": model.n_positive,
        "n_negative": model.n_negative,
        "n_weak": model.n_weak,
        "intercept": model.intercept,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _index_file_lock(path.parent).acquire():
        tmp = path.with_name(path.name + ".tmp")
        # Write through a file handle: np.savez appends ".npz" to bare paths.
        with open(tmp, "wb") as handle:
            np.savez(
                handle,
                mean=model.mean,
                components=model.components,
                scale=model.scale,
                coef=model.coef,
                meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
            )
        os.replace(tmp, path)


def _delete_artifact(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _load_artifact(path: Path) -> LearnedModel | None:
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta"]).decode("utf-8"))
            if int(meta.get("version", 0)) != _ARTIFACT_VERSION:
                return None
            raw_fingerprint = list(meta.get("fingerprint", (0, 0, 0))) + [0, 0, 0]
            return LearnedModel(
                mean=np.asarray(data["mean"], dtype=np.float32),
                components=np.asarray(data["components"], dtype=np.float32),
                scale=np.asarray(data["scale"], dtype=np.float32),
                coef=np.asarray(data["coef"], dtype=np.float32),
                intercept=float(meta["intercept"]),
                fingerprint=(int(raw_fingerprint[0]), int(raw_fingerprint[1]), int(raw_fingerprint[2])),
                trained_at=str(meta.get("trained_at", "")),
                n_positive=int(meta.get("n_positive", 0)),
                n_negative=int(meta.get("n_negative", 0)),
                n_weak=int(meta.get("n_weak", 0)),
            )
    except Exception:
        LOGGER.warning("Failed to load learned-ranker artifact %s (non-fatal)", path, exc_info=True)
        return None


def _load_artifact_cached(path: Path) -> LearnedModel | None:
    global _artifact_cache_key, _artifact_cache_model
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _cache_lock:
        if _artifact_cache_key == key:
            return _artifact_cache_model
    model = _load_artifact(path)
    with _cache_lock:
        _artifact_cache_key = key
        _artifact_cache_model = model
    return model


# ── math helpers (sklearn when importable, numpy fallback) ──────────────


def _fit_pca(sample: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on the corpus sample; returns (mean, components, scale)."""
    import numpy as np

    n_components = max(1, min(n_components, sample.shape[0], sample.shape[1]))
    mean = sample.mean(axis=0).astype(np.float32)
    components = None
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components, random_state=0)
        pca.fit(sample)
        mean = pca.mean_.astype(np.float32)
        components = pca.components_.astype(np.float32)
    except ImportError:
        pass
    if components is None:
        _, _, vt = np.linalg.svd(sample - mean, full_matrices=False)
        components = vt[:n_components].astype(np.float32)

    projected = (sample - mean) @ components.T
    scale = projected.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, components, scale


def _fit_logistic(
    features: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray, c_value: float
) -> tuple[np.ndarray, float]:
    import numpy as np

    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=c_value, max_iter=2000)
        clf.fit(features, labels, sample_weight=sample_weight)
        return clf.coef_[0].astype(np.float32), float(clf.intercept_[0])
    except ImportError:
        return _fit_logistic_numpy(features, labels, sample_weight, c_value)


def _fit_logistic_numpy(
    features: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray, c_value: float, iterations: int = 50
) -> tuple[np.ndarray, float]:
    """Compact IRLS (Newton) solver for L2-regularized weighted logistic regression."""
    import numpy as np

    n, d = features.shape
    design = np.hstack([features, np.ones((n, 1))])
    beta = np.zeros(d + 1, dtype=np.float64)
    lam = 1.0 / max(c_value, 1e-6)
    reg = lam * np.eye(d + 1)
    reg[-1, -1] = 0.0  # never penalize the intercept
    y = labels.astype(np.float64)
    w = sample_weight.astype(np.float64)

    for _ in range(iterations):
        z = design @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        gradient = design.T @ (w * (p - y)) + reg @ beta
        curvature = np.maximum(w * p * (1.0 - p), 1e-8)
        hessian = (design * curvature[:, None]).T @ design + reg
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = gradient * 0.01
        beta -= step
        if float(np.max(np.abs(step))) < 1e-6:
            break
    return beta[:d].astype(np.float32), float(beta[-1])


def _auc_score(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based (Mann-Whitney) AUC with tie averaging; None on a single class."""
    import numpy as np

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1

    pos_rank_sum = float(ranks[labels == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _f1_score(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    predicted = scores >= threshold
    tp = int(((labels == 1) & predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


# ── training data assembly ──────────────────────────────────────────────


def _feedback_fingerprint_counts() -> tuple[int, int]:
    from app.models import PaperFeedback, db

    count, max_id = (
        db.session.query(db.func.count(PaperFeedback.id), db.func.max(PaperFeedback.id))
        .filter(PaperFeedback.action.in_(POSITIVE_ACTIONS + NEGATIVE_ACTIONS))
        .one()
    )
    return int(count or 0), int(max_id or 0)


def _labeled_examples() -> tuple[list[tuple[int, datetime | None]], list[tuple[int, datetime | None]]]:
    """(paper_id, latest feedback created_at) per class; positives win conflicts."""
    from app.models import PaperFeedback, db

    rows = (
        db.session.query(PaperFeedback.paper_id, PaperFeedback.action, PaperFeedback.created_at)
        .filter(PaperFeedback.action.in_(POSITIVE_ACTIONS + NEGATIVE_ACTIONS))
        .order_by(PaperFeedback.created_at.asc(), PaperFeedback.id.asc())
        .all()
    )
    positive: dict[int, datetime | None] = {}
    negative: dict[int, datetime | None] = {}
    for paper_id, action, created_at in rows:
        if action in POSITIVE_ACTIONS:
            positive[paper_id] = created_at
        else:
            negative[paper_id] = created_at
    for paper_id in positive:
        negative.pop(paper_id, None)
    return sorted(positive.items()), sorted(negative.items())


def _temporal_weights(timestamps: list, now=None) -> np.ndarray:
    import numpy as np

    from app.services.text import now_utc

    now = now or now_utc()
    weights = []
    for ts in timestamps:
        age_days = 0.0
        if ts is not None:
            try:
                age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
            except TypeError:
                age_days = 0.0
        weights.append(max(0.05, 0.5 ** (age_days / RATING_HALF_LIFE_DAYS)))
    return np.asarray(weights, dtype=np.float64)


def _assemble_training_set(service, pos_items, neg_items, *, rng_seed: int = 0):
    """Build (X_raw, y, weights, pca_sample, counts) from feedback + corpus.

    Returns None when there are not enough embedded positives or no negative
    signal at all (neither explicit negatives nor unlabeled corpus papers).
    """
    import numpy as np

    pos_ids = [pid for pid, _ in pos_items]
    neg_ids = [pid for pid, _ in neg_items]
    pos_ts = dict(pos_items)
    neg_ts = dict(neg_items)

    found_pos, pos_vectors = service.get_paper_vectors(pos_ids)
    if pos_vectors.shape[0] < MIN_POSITIVE_FEEDBACK:
        return None
    found_neg, neg_vectors = service.get_paper_vectors(neg_ids)

    exclude = set(found_pos) | set(found_neg)
    weak_ids, weak_vectors = service.sample_paper_vectors(MAX_WEAK_NEGATIVES, exclude_ids=exclude, seed=rng_seed)
    if neg_vectors.shape[0] == 0 and weak_vectors.shape[0] == 0:
        return None

    pos_weights = _temporal_weights([pos_ts[pid] for pid in found_pos])
    neg_weights = _temporal_weights([neg_ts[pid] for pid in found_neg])
    n_weak = weak_vectors.shape[0]
    weak_weight_each = (
        RANDOM_NEGATIVE_RATIO * float(pos_weights.sum()) / (n_weak * NEGATIVE_WEIGHT_SCALE) if n_weak else 0.0
    )

    x_raw = np.vstack([pos_vectors, neg_vectors, weak_vectors]).astype(np.float32)
    y = np.concatenate(
        [
            np.ones(pos_vectors.shape[0]),
            np.zeros(neg_vectors.shape[0]),
            np.zeros(n_weak),
        ]
    )
    weights = np.concatenate([pos_weights, neg_weights, np.full(n_weak, weak_weight_each)])
    counts = (pos_vectors.shape[0], neg_vectors.shape[0], n_weak)
    return x_raw, y, weights, x_raw, counts


def _fit_model(x_raw, y, weights, pca_sample, counts, fingerprint) -> LearnedModel:
    import numpy as np

    from app.services.text import now_utc

    mean, components, scale = _fit_pca(pca_sample, PCA_COMPONENTS)
    features = ((x_raw - mean) @ components.T) / scale
    coef, intercept = _fit_logistic(features, y, weights, LR_C)
    return LearnedModel(
        mean=mean,
        components=components,
        scale=scale,
        coef=np.asarray(coef, dtype=np.float32),
        intercept=float(intercept),
        fingerprint=fingerprint,
        trained_at=now_utc().isoformat(),
        n_positive=counts[0],
        n_negative=counts[1],
        n_weak=counts[2],
    )


# ── public API ───────────────────────────────────────────────────────────


def _resolve_app(app):
    if app is not None:
        return app
    from flask import current_app, has_app_context

    if has_app_context():
        return current_app._get_current_object()
    return None


def _set_cache(model: LearnedModel | None, fingerprint: tuple[int, int, int] | None, state: str) -> None:
    global _cached_model, _cached_fingerprint, _cache_state
    with _cache_lock:
        _cached_model = model
        _cached_fingerprint = fingerprint
        _cache_state = state


def train_learned_ranker(app=None, *, force: bool = False) -> dict:
    """(Re)train the learned ranker if feedback/index state changed.

    Returns a status dict: ``{"available", "n_positive", "n_negative",
    "needed_positive", "trained_at", "reason"}``. Never raises — any failure
    logs and reports unavailable so feedback handling and scrapes survive.
    """
    status: dict[str, object] = {
        "available": False,
        "n_positive": 0,
        "n_negative": 0,
        "needed_positive": MIN_POSITIVE_FEEDBACK,
        "trained_at": None,
        "reason": "unavailable",
    }
    app = _resolve_app(app)
    if app is None:
        status["reason"] = "no-app"
        return status

    try:
        with app.app_context():
            pos_items, neg_items = _labeled_examples()
            status["n_positive"] = len(pos_items)
            status["n_negative"] = len(neg_items)
            status["needed_positive"] = max(0, MIN_POSITIVE_FEEDBACK - len(pos_items))
            try:
                learned_preferences(app.config.get("SCRAPER_CONFIG"))
            except Exception:  # pragma: no cover - snapshot refresh is best-effort
                pass

            if len(pos_items) < MIN_POSITIVE_FEEDBACK:
                # Cold start: cheap DB-only exit, no embedding service needed.
                _set_cache(None, None, "unavailable")
                _delete_artifact(_artifact_path())
                status["reason"] = "not-enough-feedback"
                return status

            from app.services.embeddings import get_embedding_service

            service = get_embedding_service(app)
            count, max_id = _feedback_fingerprint_counts()
            fingerprint = (count, max_id, service.index_size())

            with _cache_lock:
                if not force and _cached_fingerprint == fingerprint and _cache_state != "unknown":
                    model = _cached_model
                    status["available"] = model is not None
                    status["trained_at"] = model.trained_at if model else None
                    status["reason"] = "cached" if model else "not-enough-embeddings"
                    return status

            training_set = _assemble_training_set(service, pos_items, neg_items)
            if training_set is None:
                _set_cache(None, fingerprint, "unavailable")
                status["reason"] = "not-enough-embeddings"
                return status

            x_raw, y, weights, pca_sample, counts = training_set
            model = _fit_model(x_raw, y, weights, pca_sample, counts, fingerprint)
            _set_cache(model, fingerprint, "available")
            artifact = _artifact_path()
            if artifact is not None:
                try:
                    _save_artifact(model, artifact)
                except Exception:
                    LOGGER.warning("Persisting learned-ranker artifact failed (non-fatal)", exc_info=True)

            status["available"] = True
            status["trained_at"] = model.trained_at
            status["reason"] = "trained"
            LOGGER.info(
                "Learned ranker retrained: %d positives, %d negatives, %d weak negatives",
                model.n_positive,
                model.n_negative,
                model.n_weak,
            )

            try:
                evaluate_learned_ranker(app)
            except Exception:
                LOGGER.warning("Learned-ranker evaluation failed (non-fatal)", exc_info=True)
            return status
    except Exception:
        LOGGER.warning("Learned-ranker training failed (non-fatal)", exc_info=True)
        status["reason"] = "error"
        return status


def ensure_learned_model(app=None) -> LearnedModel | None:
    """Train if stale (fingerprint-cached) and return the active model."""
    train_learned_ranker(app)
    with _cache_lock:
        return _cached_model


def peek_learned_model() -> LearnedModel | None:
    """Return the trained model WITHOUT touching the DB or training.

    Uses the in-process cache when this process has trained/decided already,
    else falls back to the on-disk artifact (kept fresh by the feedback
    retrain hook, possibly by another process). Cheap enough per paper row.
    """
    with _cache_lock:
        if _cache_state == "available":
            return _cached_model
        if _cache_state == "unavailable":
            return None
    path = _artifact_path()
    if path is None:
        return None
    return _load_artifact_cached(path)


def score_vectors(vectors, model: LearnedModel | None = None):
    """Probabilities in [0, 1] for (n, 768) embedding vectors.

    Returns None when no model is available.
    """
    model = model if model is not None else peek_learned_model()
    if model is None:
        return None
    return model.predict_proba(vectors)


def interest_signal(
    vector,
    profile: InterestProfile | None,
    model: LearnedModel | None,
    blend: float,
) -> tuple[float | None, str | None]:
    """Blended interest signal in [-1, 1] plus its source label.

    LR probability is mapped to [-1, 1] so ``interest_weight`` keeps its
    existing semantics. With no model the centroid similarity passes through
    unchanged (source "centroid"); with no profile the LR signal is used alone.
    """
    centroid_score = None
    if profile is not None:
        from app.services.interest_model import score_vector

        centroid_score = score_vector(profile, vector)

    lr_signal = None
    if model is not None:
        import numpy as np

        probability = float(model.predict_proba(np.asarray(vector, dtype=np.float32))[0])
        lr_signal = 2.0 * probability - 1.0

    if lr_signal is None and centroid_score is None:
        return None, None
    if lr_signal is None:
        return centroid_score, "centroid"
    if centroid_score is None:
        return lr_signal, "learned"
    blend = min(max(float(blend), 0.0), 1.0)
    return blend * lr_signal + (1.0 - blend) * centroid_score, "learned"


def resolve_interest_source(config: dict | None) -> str:
    """Honest label for the interest score component: "learned" or "centroid"."""
    try:
        prefs = learned_preferences(config)
        if prefs.get("enabled", True) and peek_learned_model() is not None:
            return "learned"
    except Exception:  # pragma: no cover - label resolution must never break rendering
        pass
    return "centroid"


def model_status(app=None) -> dict:
    """Status snapshot for the settings UI. Never raises."""
    status: dict[str, object] = {
        "enabled": True,
        "available": False,
        "n_positive": 0,
        "n_negative": 0,
        "needed_positive": MIN_POSITIVE_FEEDBACK,
        "trained_at": None,
        "last_auc": None,
        "last_f1": None,
    }
    app = _resolve_app(app)
    if app is None:
        return status
    try:
        with app.app_context():
            prefs = learned_preferences(app.config.get("SCRAPER_CONFIG"))
            status["enabled"] = bool(prefs.get("enabled", True))
            train_status = train_learned_ranker(app)
            status.update(
                {
                    "available": bool(train_status["available"]) and status["enabled"],
                    "n_positive": train_status["n_positive"],
                    "n_negative": train_status["n_negative"],
                    "needed_positive": train_status["needed_positive"],
                    "trained_at": train_status["trained_at"],
                }
            )

            from app.models import RecommendationMetric

            for metric_name, key in (("learned_ranker_auc", "last_auc"), ("learned_ranker_f1", "last_f1")):
                row = (
                    RecommendationMetric.query.filter_by(metric_name=metric_name)
                    .order_by(RecommendationMetric.measured_at.desc(), RecommendationMetric.id.desc())
                    .first()
                )
                if row is not None:
                    status[key] = float(row.metric_value)
    except Exception:
        LOGGER.warning("Learned-ranker status lookup failed (non-fatal)", exc_info=True)
    return status


def evaluate_learned_ranker(app=None) -> dict | None:
    """Offline eval: AUC + F1 on a time-ordered holdout of the user's feedback.

    Trains the recipe on the earliest 80% of labeled feedback and scores the
    most recent 20%. Writes ``learned_ranker_auc`` / ``learned_ranker_f1``
    rows to RecommendationMetric (log only — no UI yet). Returns the metrics
    dict, or None when there is not enough two-class data to evaluate.
    """
    import numpy as np

    app = _resolve_app(app)
    if app is None:
        return None

    with app.app_context():
        pos_items, neg_items = _labeled_examples()
        labeled = [(pid, ts, 1) for pid, ts in pos_items] + [(pid, ts, 0) for pid, ts in neg_items]
        if len(labeled) < 8:
            return None
        # created_at is naive (SQLite server default); None sorts first.
        labeled.sort(key=lambda item: item[1] or datetime.min)
        split = int(len(labeled) * 0.8)
        train_part, test_part = labeled[:split], labeled[split:]

        train_pos = [(pid, ts) for pid, ts, label in train_part if label == 1]
        train_neg = [(pid, ts) for pid, ts, label in train_part if label == 0]
        if len(train_pos) < 3 or len({label for _, _, label in test_part}) < 2:
            return None

        from app.services.embeddings import get_embedding_service

        service = get_embedding_service(app)
        training_set = _assemble_training_set(service, train_pos, train_neg, rng_seed=1)
        if training_set is None:
            return None
        x_raw, y, weights, pca_sample, counts = training_set
        model = _fit_model(x_raw, y, weights, pca_sample, counts, (0, 0, 0))

        test_ids = [pid for pid, _, _ in test_part]
        label_by_id = {pid: label for pid, _, label in test_part}
        found_ids, test_vectors = service.get_paper_vectors(test_ids)
        if len(found_ids) < 2:
            return None
        labels = np.asarray([label_by_id[pid] for pid in found_ids])
        if len(set(labels.tolist())) < 2:
            return None

        scores = model.predict_proba(test_vectors)
        auc = _auc_score(labels, scores)
        if auc is None:
            return None
        f1 = _f1_score(labels, scores)

        from app.models import RecommendationMetric, db

        snapshot = {
            "source": "learned_ranker_eval",
            "n_train": len(train_part),
            "n_test": int(len(found_ids)),
        }
        db.session.add(
            RecommendationMetric(metric_name="learned_ranker_auc", metric_value=float(auc), config_snapshot=snapshot)
        )
        db.session.add(
            RecommendationMetric(metric_name="learned_ranker_f1", metric_value=float(f1), config_snapshot=snapshot)
        )
        db.session.commit()
        LOGGER.info("Learned-ranker holdout eval: AUC=%.3f F1=%.3f (n_test=%d)", auc, f1, len(found_ids))
        return {"auc": float(auc), "f1": float(f1), "n_test": int(len(found_ids))}


# ── retrain-on-feedback hook ─────────────────────────────────────────────


def request_retrain(app=None) -> None:
    """Schedule a debounced retrain after a feedback event. Never raises.

    Bulk feedback (N rapid events) collapses into one retrain: each call
    restarts a short trailing timer. Under TESTING the retrain runs
    synchronously so tests stay deterministic and no timers leak across cases.
    """
    global _pending_retrain
    try:
        app = _resolve_app(app)
        if app is None:
            return
        try:
            with app.app_context():
                if not learned_preferences(app.config.get("SCRAPER_CONFIG")).get("enabled", True):
                    return
        except Exception:  # pragma: no cover - preference lookup is best-effort
            pass
        if app.config.get("TESTING"):
            train_learned_ranker(app)
            return
        with _retrain_lock:
            if _pending_retrain is not None:
                _pending_retrain.cancel()
            timer = threading.Timer(RETRAIN_DEBOUNCE_SECONDS, _run_scheduled_retrain, args=(app,))
            timer.daemon = True
            _pending_retrain = timer
            timer.start()
    except Exception:
        LOGGER.warning("Scheduling learned-ranker retrain failed (non-fatal)", exc_info=True)


def _run_scheduled_retrain(app) -> None:
    global _pending_retrain
    with _retrain_lock:
        _pending_retrain = None
    try:
        train_learned_ranker(app)
    except Exception:  # pragma: no cover - train already swallows its errors
        LOGGER.warning("Scheduled learned-ranker retrain failed (non-fatal)", exc_info=True)


def flush_pending_retrain() -> bool:
    """Run any pending debounced retrain now (tests/shutdown). True if one ran."""
    global _pending_retrain
    with _retrain_lock:
        timer = _pending_retrain
        _pending_retrain = None
    if timer is None:
        return False
    timer.cancel()
    timer.function(*timer.args, **timer.kwargs)
    return True


def cancel_pending_retrain() -> None:
    global _pending_retrain
    with _retrain_lock:
        if _pending_retrain is not None:
            _pending_retrain.cancel()
            _pending_retrain = None


def reset_learned_ranker_cache() -> None:
    """Reset all module state (for testing)."""
    global _artifact_cache_key, _artifact_cache_model, _runtime_prefs
    cancel_pending_retrain()
    _set_cache(None, None, "unknown")
    with _cache_lock:
        _artifact_cache_key = None
        _artifact_cache_model = None
    with _runtime_prefs_lock:
        _runtime_prefs = None
