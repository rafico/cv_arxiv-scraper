#!/usr/bin/env python3
"""Offline benchmark of the learned-ranker recipe on the Scholar Inbox dataset.

Runs the *production* training pipeline (``_assemble_training_set`` →
``_fit_model`` from ``app.services.learned_ranker``) per dataset user on a
time-ordered 80/20 split, and reports AUC / nDCG@10 / recall@20 / MRR — the
same metrics the in-app holdout eval writes to RecommendationMetric.

Dataset: the public ~800k-rating corpus released with the Scholar Inbox paper
(arXiv:2504.08385). It is never bundled here — download it separately and pass
``--data``. Expected columns (rename via the ``--col-*`` flags): ``user_id``,
``rating`` (positive when >= --positive-threshold), ``timestamp`` (ISO 8601 or
epoch seconds; optional), ``title``, ``abstract``. CSV is read with the stdlib;
``.parquet`` needs pandas+pyarrow installed.

Embeddings are computed with the app's embedding model (SPECTER2 with
fallbacks) and cached per abstract hash in ``--cache``, so re-runs are cheap.

No dataset at hand? ``--self-test`` runs the same per-user evaluation loop on
synthetic separable vectors and asserts the recipe achieves AUC ≈ 1.0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.services.learned_ranker import (  # noqa: E402
    _assemble_training_set,
    _auc_score,
    _fit_model,
    _mrr,
    _ndcg_at_k,
    _recall_at_k,
)


class _ArrayService:
    """In-memory stand-in for EmbeddingService's vector-lookup surface.

    ``_assemble_training_set`` only needs ``get_paper_vectors`` and
    ``sample_paper_vectors``; backing them with a plain dict lets the benchmark
    reuse the production training-set assembly (weak negatives, temporal
    weights) without a Flask app or FAISS index.
    """

    def __init__(self, vectors_by_id: dict[int, np.ndarray]):
        self._vectors = vectors_by_id

    def get_paper_vectors(self, paper_ids: list[int]) -> tuple[list[int], np.ndarray]:
        found = [pid for pid in paper_ids if pid in self._vectors]
        if not found:
            return [], np.empty((0, 0), dtype=np.float32)
        return found, np.vstack([self._vectors[pid] for pid in found]).astype(np.float32)

    def sample_paper_vectors(self, count: int, exclude_ids=None, seed: int = 0) -> tuple[list[int], np.ndarray]:
        exclude = set(exclude_ids or ())
        pool = sorted(pid for pid in self._vectors if pid not in exclude)
        rng = np.random.default_rng(seed)
        if len(pool) > count:
            pool = list(rng.choice(pool, size=count, replace=False))
        if not pool:
            return [], np.empty((0, 0), dtype=np.float32)
        return pool, np.vstack([self._vectors[pid] for pid in pool]).astype(np.float32)


def _parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    except (ValueError, OSError):
        pass
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _load_rows(path: Path, cols: dict[str, str]) -> list[dict]:
    """Normalize the dataset to [{user, positive?, ts, text}] rows."""
    if path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError:
            sys.exit("Reading .parquet needs pandas+pyarrow: pip install pandas pyarrow")
        records = pd.read_parquet(path).to_dict("records")
        raw_rows = ({k: ("" if v is None else str(v)) for k, v in rec.items()} for rec in records)
    else:
        raw_rows = csv.DictReader(path.open(newline="", encoding="utf-8"))

    rows = []
    for raw in raw_rows:
        title = (raw.get(cols["title"]) or "").strip()
        abstract = (raw.get(cols["abstract"]) or "").strip()
        rating_raw = (raw.get(cols["rating"]) or "").strip()
        if not (title or abstract) or not rating_raw:
            continue
        try:
            rating = float(rating_raw)
        except ValueError:
            continue
        rows.append(
            {
                "user": (raw.get(cols["user"]) or "").strip(),
                "rating": rating,
                "ts": _parse_timestamp((raw.get(cols["time"]) or "").strip()),
                "text": f"{title}. {abstract}".strip(". "),
            }
        )
    return rows


def _embed_texts(texts: list[str], cache_dir: Path) -> dict[str, np.ndarray]:
    """text -> unit vector, cached by content hash so re-runs skip the model."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "embeddings.npz"
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path) as archive:
            cached = {key: archive[key] for key in archive.files}

    keyed = {hashlib.sha1(text.encode()).hexdigest(): text for text in texts}  # noqa: S324 - cache key, not security
    missing = [key for key in keyed if key not in cached]
    if missing:
        from app.services.embeddings import EmbeddingService

        service = EmbeddingService(cache_dir / "model_index")
        batch = 64
        for start in range(0, len(missing), batch):
            keys = missing[start : start + batch]
            vectors = service.encode([keyed[key] for key in keys])
            cached.update(dict(zip(keys, vectors)))
            print(f"  embedded {min(start + batch, len(missing))}/{len(missing)} new abstracts", flush=True)
        np.savez_compressed(cache_path, **cached)

    return {text: cached[key] for key, text in keyed.items()}


def _evaluate_user(rows: list[dict], vectors_by_id: dict[int, np.ndarray], seed: int) -> dict | None:
    """Time-ordered 80/20 split → production training → holdout ranking metrics."""
    rows = sorted(rows, key=lambda row: row["ts"] or datetime.min.replace(tzinfo=timezone.utc))
    split = int(len(rows) * 0.8)
    train_part, test_part = rows[:split], rows[split:]

    train_pos = [(row["pid"], row["ts"]) for row in train_part if row["positive"]]
    train_neg = [(row["pid"], row["ts"]) for row in train_part if not row["positive"]]
    test_ids = [row["pid"] for row in test_part]
    if len(train_pos) < 3 or len({row["positive"] for row in test_part}) < 2:
        return None

    service = _ArrayService(vectors_by_id)
    training_set = _assemble_training_set(service, train_pos, train_neg, rng_seed=seed, extra_exclude=test_ids)
    if training_set is None:
        return None
    model = _fit_model(*training_set, (0, 0, 0))

    label_by_id = {row["pid"]: int(row["positive"]) for row in test_part}
    found_ids, test_vectors = service.get_paper_vectors(test_ids)
    labels = np.asarray([label_by_id[pid] for pid in found_ids])
    if len(found_ids) < 2 or len(set(labels.tolist())) < 2:
        return None

    scores = model.predict_proba(test_vectors)
    auc = _auc_score(labels, scores)
    if auc is None:
        return None
    return {
        "auc": float(auc),
        "ndcg10": float(_ndcg_at_k(labels, scores, k=10)),
        "recall20": float(_recall_at_k(labels, scores, k=20)),
        "mrr": float(_mrr(labels, scores)),
        "n_test": int(len(found_ids)),
    }


def _summarize(per_user: list[dict]) -> dict:
    summary: dict[str, object] = {"n_users": len(per_user)}
    for metric in ("auc", "ndcg10", "recall20", "mrr"):
        values = [user[metric] for user in per_user]
        summary[metric] = {
            "mean": round(statistics.fmean(values), 4),
            "median": round(statistics.median(values), 4),
        }
    return summary


def self_test(seed: int = 0) -> dict:
    """The full per-user loop on synthetic separable clusters; AUC must be ~1."""
    rng = np.random.default_rng(seed)

    def _unit(center: np.ndarray) -> np.ndarray:
        vec = center + rng.normal(scale=0.05, size=center.shape)
        return (vec / np.linalg.norm(vec)).astype(np.float32)

    dim = 768
    liked, disliked = rng.normal(size=dim), rng.normal(size=dim)
    vectors_by_id: dict[int, np.ndarray] = {}
    rows = []
    for idx in range(60):
        positive = idx % 2 == 0
        vectors_by_id[idx] = _unit(liked if positive else disliked)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        rows.append({"pid": idx, "positive": positive, "ts": ts})
    # Unlabeled corpus papers for the weak-negative sampler.
    for idx in range(1000, 1200):
        vectors_by_id[idx] = _unit(rng.normal(size=dim))

    result = _evaluate_user(rows, vectors_by_id, seed=seed)
    assert result is not None, "self-test produced no evaluable holdout"
    assert result["auc"] > 0.95, f"recipe failed to separate synthetic clusters: {result}"
    assert result["ndcg10"] > 0.9, f"ranking metrics off on separable data: {result}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="dataset file (.csv or .parquet)")
    parser.add_argument("--cache", type=Path, default=Path(".benchmark_cache"), help="embedding cache dir")
    parser.add_argument("--min-ratings", type=int, default=20, help="skip users with fewer ratings")
    parser.add_argument("--max-users", type=int, default=0, help="evaluate at most N users (0 = all)")
    parser.add_argument("--positive-threshold", type=float, default=1.0, help="rating >= this is a positive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="write the full report to this path")
    parser.add_argument("--self-test", action="store_true", help="run on synthetic data, no dataset needed")
    for name, default in (
        ("user", "user_id"),
        ("rating", "rating"),
        ("time", "timestamp"),
        ("title", "title"),
        ("abstract", "abstract"),
    ):
        parser.add_argument(f"--col-{name}", default=default, dest=f"col_{name}")
    args = parser.parse_args()

    if args.self_test:
        result = self_test(args.seed)
        print(f"self-test OK: {result}")
        return
    if not args.data:
        parser.error("--data is required (or use --self-test)")

    cols = {name: getattr(args, f"col_{name}") for name in ("user", "rating", "time", "title", "abstract")}
    rows = _load_rows(args.data, cols)
    print(f"loaded {len(rows)} ratings from {args.data}")

    text_vectors = _embed_texts(sorted({row["text"] for row in rows}), args.cache)
    text_ids = {text: pid for pid, text in enumerate(sorted(text_vectors))}
    vectors_by_id = {text_ids[text]: vec for text, vec in text_vectors.items()}

    by_user: dict[str, list[dict]] = {}
    for row in rows:
        by_user.setdefault(row["user"], []).append(
            {"pid": text_ids[row["text"]], "positive": row["rating"] >= args.positive_threshold, "ts": row["ts"]}
        )

    per_user = []
    eligible = [user for user, user_rows in sorted(by_user.items()) if len(user_rows) >= args.min_ratings]
    if args.max_users:
        eligible = eligible[: args.max_users]
    for index, user in enumerate(eligible, start=1):
        result = _evaluate_user(by_user[user], vectors_by_id, seed=args.seed)
        if result is not None:
            per_user.append({"user": user, **result})
        if index % 50 == 0:
            print(f"  evaluated {index}/{len(eligible)} users", flush=True)

    if not per_user:
        sys.exit("no user had enough two-class, timestamped data to evaluate")
    summary = _summarize(per_user)
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "per_user": per_user}, indent=2))
        print(f"full report written to {args.json}")


if __name__ == "__main__":
    main()
