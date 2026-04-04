"""Corpus-level clustering and similarity helpers backed by FAISS embeddings."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import timedelta
from math import log

import numpy as np
from sqlalchemy import and_, or_

from app.models import Paper
from app.services.matching import check_author_match
from app.services.text import STOP_WORDS, now_utc, tokenize

LOGGER = logging.getLogger(__name__)

DEFAULT_CLUSTER_LIMIT = 200
DEFAULT_CLUSTER_SAMPLE_LIMIT = 5
DEFAULT_EMERGING_SAMPLE_LIMIT = 3
DEFAULT_NEIGHBOR_LIMIT = 20


def _paper_tokens(paper: Paper) -> list[str]:
    text = " ".join(filter(None, [paper.title or "", paper.abstract_text or ""]))
    return [token for token in tokenize(text) if token not in STOP_WORDS]


def _serialize_paper(
    paper: Paper,
    *,
    similarity_score: float | None = None,
    centroid_similarity: float | None = None,
    matched_seed_ids: list[int] | None = None,
    tracked_author_matches: list[str] | None = None,
) -> dict:
    payload = {
        "id": paper.id,
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.authors,
        "paper_score": float(paper.paper_score or 0.0),
        "publication_dt": paper.publication_dt.isoformat() if paper.publication_dt else None,
        "scraped_at": paper.scraped_at.isoformat() if paper.scraped_at else None,
    }
    if similarity_score is not None:
        payload["similarity_score"] = round(float(similarity_score), 4)
    if centroid_similarity is not None:
        payload["centroid_similarity"] = round(float(centroid_similarity), 4)
    if matched_seed_ids is not None:
        payload["matched_seed_ids"] = matched_seed_ids
    if tracked_author_matches:
        payload["tracked_author_matches"] = tracked_author_matches
    return payload


def _resolve_embedding_service(embedding_service=None):
    if embedding_service is not None:
        return embedding_service
    from app.services.embeddings import get_embedding_service

    return get_embedding_service()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _resolve_cluster_count(item_count: int, requested: int | None = None) -> int:
    if item_count <= 1:
        return item_count
    if requested is not None:
        return max(1, min(requested, item_count))
    heuristic = int(round(np.sqrt(item_count)))
    return max(1, min(8, heuristic or 1, item_count))


def _run_kmeans(
    vectors: np.ndarray,
    cluster_count: int,
    *,
    max_iter: int = 25,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    if len(vectors) == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0, 0), dtype=np.float32)

    cluster_count = max(1, min(cluster_count, len(vectors)))
    vectors = _normalize_rows(vectors)

    if cluster_count == 1:
        centroid = _normalize_rows(vectors.mean(axis=0, keepdims=True))
        labels = np.zeros(len(vectors), dtype=np.int32)
        return labels, centroid

    rng = np.random.default_rng(seed)
    centroid_indices = rng.choice(len(vectors), size=cluster_count, replace=False)
    centroids = vectors[centroid_indices].copy()
    labels = np.full(len(vectors), -1, dtype=np.int32)

    for _ in range(max_iter):
        similarities = vectors @ centroids.T
        next_labels = similarities.argmax(axis=1).astype(np.int32)

        if np.array_equal(labels, next_labels):
            labels = next_labels
            break

        labels = next_labels
        updated_centroids: list[np.ndarray] = []
        for cluster_id in range(cluster_count):
            member_vectors = vectors[labels == cluster_id]
            if len(member_vectors) == 0:
                updated_centroids.append(vectors[rng.integers(len(vectors))])
                continue
            updated_centroids.append(_normalize_rows(member_vectors.mean(axis=0, keepdims=True))[0])
        centroids = np.asarray(updated_centroids, dtype=np.float32)

    return labels, centroids


def _label_cluster(
    papers: list[Paper],
    *,
    token_cache: dict[int, list[str]],
    corpus_doc_freq: Counter[str],
    corpus_doc_count: int,
    max_terms: int = 3,
) -> str:
    term_counts: Counter[str] = Counter()
    for paper in papers:
        term_counts.update(token_cache.get(paper.id, []))

    scored_terms: list[tuple[float, str]] = []
    for term, count in term_counts.items():
        idf = log((1 + corpus_doc_count) / (1 + corpus_doc_freq.get(term, 0))) + 1.0
        scored_terms.append((count * idf, term))

    scored_terms.sort(key=lambda item: (-item[0], item[1]))
    labels = [term for _, term in scored_terms[:max_terms]]
    return ", ".join(labels) if labels else "miscellaneous"


def _window_bounds(window_days: int, *, offset_days: int = 0, reference_time=None) -> tuple:
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if offset_days < 0:
        raise ValueError("offset_days cannot be negative")
    end_dt = reference_time or now_utc()
    if offset_days:
        end_dt = end_dt - timedelta(days=offset_days)
    start_dt = end_dt - timedelta(days=window_days)
    return start_dt, end_dt


def _papers_in_window(
    window_days: int,
    *,
    offset_days: int = 0,
    limit: int = DEFAULT_CLUSTER_LIMIT,
    reference_time=None,
) -> list[Paper]:
    start_dt, end_dt = _window_bounds(window_days, offset_days=offset_days, reference_time=reference_time)
    start_date = start_dt.date()
    end_date = end_dt.date()

    query = Paper.query.filter(Paper.is_hidden.is_(False)).filter(
        or_(
            and_(
                Paper.publication_dt.isnot(None),
                Paper.publication_dt > start_date,
                Paper.publication_dt <= end_date,
            ),
            and_(
                Paper.publication_dt.is_(None),
                Paper.scraped_at > start_dt,
                Paper.scraped_at <= end_dt,
            ),
        )
    )

    query = query.order_by(Paper.scraped_at.desc(), Paper.paper_score.desc())
    if limit:
        query = query.limit(limit)
    return query.all()


def _cluster_corpus(
    papers: list[Paper],
    *,
    cluster_count: int | None = None,
    paper_limit: int = DEFAULT_CLUSTER_SAMPLE_LIMIT,
    embedding_service=None,
) -> dict:
    if not papers:
        return {
            "paper_count": 0,
            "indexed_paper_count": 0,
            "cluster_count": 0,
            "clusters": [],
            "assignments": {},
        }

    paper_limit = max(1, paper_limit)
    ordered_ids = [paper.id for paper in papers]
    indexed_ids, vectors = embedding_service.get_paper_vectors(ordered_ids)
    if not indexed_ids:
        return {
            "paper_count": len(papers),
            "indexed_paper_count": 0,
            "cluster_count": 0,
            "clusters": [],
            "assignments": {},
        }

    vectors = _normalize_rows(vectors)
    cluster_total = _resolve_cluster_count(len(indexed_ids), requested=cluster_count)
    labels, centroids = _run_kmeans(vectors, cluster_total)

    papers_by_id = {paper.id: paper for paper in papers}
    clustered_papers = [papers_by_id[paper_id] for paper_id in indexed_ids if paper_id in papers_by_id]
    token_cache = {paper.id: _paper_tokens(paper) for paper in clustered_papers}
    corpus_doc_freq: Counter[str] = Counter()
    for tokens in token_cache.values():
        corpus_doc_freq.update(set(tokens))

    assignments: dict[int, int] = {}
    member_indices_by_cluster: dict[int, list[int]] = defaultdict(list)
    for idx, cluster_id in enumerate(labels.tolist()):
        paper_id = indexed_ids[idx]
        assignments[paper_id] = cluster_id
        member_indices_by_cluster[cluster_id].append(idx)

    clusters: list[dict] = []
    for cluster_id, member_indices in member_indices_by_cluster.items():
        centroid = centroids[cluster_id]
        ranked_members: list[tuple[int, float]] = []
        for idx in member_indices:
            similarity = float(vectors[idx] @ centroid)
            ranked_members.append((idx, similarity))

        ranked_members.sort(
            key=lambda item: (
                -item[1],
                -(clustered_papers[item[0]].paper_score or 0.0),
                clustered_papers[item[0]].title or "",
            )
        )

        member_papers = [clustered_papers[idx] for idx, _ in ranked_members]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "label": _label_cluster(
                    member_papers,
                    token_cache=token_cache,
                    corpus_doc_freq=corpus_doc_freq,
                    corpus_doc_count=len(clustered_papers),
                ),
                "size": len(member_papers),
                "paper_ids": [paper.id for paper in member_papers],
                "papers": [
                    _serialize_paper(clustered_papers[idx], centroid_similarity=similarity)
                    for idx, similarity in ranked_members[:paper_limit]
                ],
                "mean_similarity": round(float(np.mean([similarity for _, similarity in ranked_members])), 4),
            }
        )

    clusters.sort(key=lambda item: (-item["size"], item["label"], item["cluster_id"]))
    return {
        "paper_count": len(papers),
        "indexed_paper_count": len(indexed_ids),
        "cluster_count": len(clusters),
        "clusters": clusters,
        "assignments": assignments,
    }


def analyze_topic_clusters(
    *,
    window_days: int = 7,
    offset_days: int = 0,
    limit: int = DEFAULT_CLUSTER_LIMIT,
    cluster_count: int | None = None,
    paper_limit: int = DEFAULT_CLUSTER_SAMPLE_LIMIT,
    reference_time=None,
    embedding_service=None,
) -> dict:
    papers = _papers_in_window(window_days, offset_days=offset_days, limit=limit, reference_time=reference_time)
    try:
        service = _resolve_embedding_service(embedding_service)
    except Exception as exc:
        LOGGER.warning("Corpus clustering unavailable: %s", exc)
        return {
            "window_days": window_days,
            "offset_days": offset_days,
            "paper_count": len(papers),
            "indexed_paper_count": 0,
            "cluster_count": 0,
            "clusters": [],
        }

    result = _cluster_corpus(
        papers,
        cluster_count=cluster_count,
        paper_limit=paper_limit,
        embedding_service=service,
    )
    result.pop("assignments", None)
    result["window_days"] = window_days
    result["offset_days"] = offset_days
    return result


def detect_emerging_topics(
    *,
    recent_days: int = 7,
    baseline_days: int = 28,
    limit: int = DEFAULT_CLUSTER_LIMIT,
    cluster_count: int | None = None,
    paper_limit: int = DEFAULT_EMERGING_SAMPLE_LIMIT,
    reference_time=None,
    embedding_service=None,
) -> dict:
    recent_papers = _papers_in_window(recent_days, limit=limit, reference_time=reference_time)
    baseline_papers = _papers_in_window(
        baseline_days,
        offset_days=recent_days,
        limit=limit,
        reference_time=reference_time,
    )

    combined_by_id = {paper.id: paper for paper in baseline_papers}
    combined_by_id.update({paper.id: paper for paper in recent_papers})

    try:
        service = _resolve_embedding_service(embedding_service)
    except Exception as exc:
        LOGGER.warning("Emerging topic analysis unavailable: %s", exc)
        return {
            "recent_days": recent_days,
            "baseline_days": baseline_days,
            "recent_paper_count": len(recent_papers),
            "baseline_paper_count": len(baseline_papers),
            "indexed_recent_paper_count": 0,
            "indexed_baseline_paper_count": 0,
            "topics": [],
        }

    clustered = _cluster_corpus(
        list(combined_by_id.values()),
        cluster_count=cluster_count,
        paper_limit=max(paper_limit, DEFAULT_CLUSTER_SAMPLE_LIMIT),
        embedding_service=service,
    )
    assignments = clustered.pop("assignments", {})
    recent_ids = {paper.id for paper in recent_papers if paper.id in assignments}
    baseline_ids = {paper.id for paper in baseline_papers if paper.id in assignments}

    if not recent_ids:
        return {
            "recent_days": recent_days,
            "baseline_days": baseline_days,
            "recent_paper_count": len(recent_papers),
            "baseline_paper_count": len(baseline_papers),
            "indexed_recent_paper_count": 0,
            "indexed_baseline_paper_count": len(baseline_ids),
            "topics": [],
        }

    counts: dict[int, dict[str, int]] = defaultdict(lambda: {"recent": 0, "baseline": 0})
    for paper_id in recent_ids:
        counts[assignments[paper_id]]["recent"] += 1
    for paper_id in baseline_ids:
        counts[assignments[paper_id]]["baseline"] += 1

    cluster_map = {cluster["cluster_id"]: cluster for cluster in clustered["clusters"]}
    topics: list[dict] = []
    recent_total = len(recent_ids)
    baseline_total = len(baseline_ids)
    for cluster_id, cluster in cluster_map.items():
        recent_count = counts[cluster_id]["recent"]
        if recent_count == 0:
            continue

        baseline_count = counts[cluster_id]["baseline"]
        recent_share = recent_count / recent_total if recent_total else 0.0
        baseline_share = baseline_count / baseline_total if baseline_total else 0.0
        delta_share = recent_share - baseline_share
        if delta_share <= 0:
            continue

        recent_samples = [
            _serialize_paper(combined_by_id[paper_id])
            for paper_id in cluster["paper_ids"]
            if paper_id in recent_ids and paper_id in combined_by_id
        ][:paper_limit]

        topics.append(
            {
                "cluster_id": cluster_id,
                "label": cluster["label"],
                "size": cluster["size"],
                "recent_count": recent_count,
                "baseline_count": baseline_count,
                "recent_share": round(recent_share, 4),
                "baseline_share": round(baseline_share, 4),
                "delta_share": round(delta_share, 4),
                "recent_papers": recent_samples,
            }
        )

    topics.sort(key=lambda item: (-item["delta_share"], -item["recent_count"], item["label"]))
    return {
        "recent_days": recent_days,
        "baseline_days": baseline_days,
        "recent_paper_count": len(recent_papers),
        "baseline_paper_count": len(baseline_papers),
        "indexed_recent_paper_count": len(recent_ids),
        "indexed_baseline_paper_count": len(baseline_ids),
        "topics": topics,
    }


def find_neighbor_papers(
    seed_paper_ids: list[int],
    *,
    limit: int = DEFAULT_NEIGHBOR_LIMIT,
    tracked_authors: list[str] | None = None,
    exclude_tracked_authors: bool = True,
    embedding_service=None,
) -> dict:
    deduped_seed_ids = list(dict.fromkeys(int(paper_id) for paper_id in seed_paper_ids if paper_id))
    tracked_authors = [author for author in (tracked_authors or []) if author]
    if not deduped_seed_ids:
        return {
            "seed_paper_ids": [],
            "excluded_tracked_authors": exclude_tracked_authors,
            "tracked_author_count": len(tracked_authors),
            "results": [],
        }

    try:
        service = _resolve_embedding_service(embedding_service)
    except Exception as exc:
        LOGGER.warning("Neighbor search unavailable: %s", exc)
        return {
            "seed_paper_ids": deduped_seed_ids,
            "excluded_tracked_authors": exclude_tracked_authors,
            "tracked_author_count": len(tracked_authors),
            "results": [],
        }

    candidate_limit = max(limit * 5, 25)
    seed_id_set = set(deduped_seed_ids)
    aggregated: dict[int, dict] = {}
    for seed_paper_id in deduped_seed_ids:
        try:
            results = service.search_by_id(seed_paper_id, top_k=candidate_limit)
        except Exception as exc:
            LOGGER.debug("Neighbor lookup failed for paper %s: %s", seed_paper_id, exc)
            continue

        for paper_id, score in results:
            if paper_id in seed_id_set:
                continue
            entry = aggregated.setdefault(
                paper_id,
                {
                    "similarity_score": float(score),
                    "matched_seed_ids": [],
                },
            )
            entry["similarity_score"] = max(entry["similarity_score"], float(score))
            if seed_paper_id not in entry["matched_seed_ids"]:
                entry["matched_seed_ids"].append(seed_paper_id)

    if not aggregated:
        return {
            "seed_paper_ids": deduped_seed_ids,
            "excluded_tracked_authors": exclude_tracked_authors,
            "tracked_author_count": len(tracked_authors),
            "results": [],
        }

    papers = Paper.query.filter(Paper.id.in_(aggregated.keys())).all()
    papers_by_id = {paper.id: paper for paper in papers}

    scored_results: list[dict] = []
    for paper_id, meta in aggregated.items():
        paper = papers_by_id.get(paper_id)
        if paper is None or paper.is_hidden:
            continue

        tracked_matches = check_author_match(
            [name.strip() for name in paper.authors.split(",") if name.strip()],
            tracked_authors,
        )
        if exclude_tracked_authors and tracked_matches:
            continue

        matched_seed_ids = sorted(meta["matched_seed_ids"])
        scored_results.append(
            _serialize_paper(
                paper,
                similarity_score=meta["similarity_score"],
                matched_seed_ids=matched_seed_ids,
                tracked_author_matches=tracked_matches,
            )
        )

    scored_results.sort(
        key=lambda item: (
            -item["similarity_score"],
            -item["paper_score"],
            item["title"],
        )
    )
    return {
        "seed_paper_ids": deduped_seed_ids,
        "excluded_tracked_authors": exclude_tracked_authors,
        "tracked_author_count": len(tracked_authors),
        "results": scored_results[:limit],
    }
