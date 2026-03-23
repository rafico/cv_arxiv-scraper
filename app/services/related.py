"""Related-paper recommendation helpers."""

from __future__ import annotations

import logging
from collections import Counter
from functools import lru_cache
from math import sqrt

from app.services.text import STOP_WORDS, tokenize

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=512)
def build_vector(text: str) -> Counter[str]:
    # Note: returns a mutable Counter cached by lru_cache — callers must not mutate.
    result = Counter(token for token in tokenize(text) if token not in STOP_WORDS)
    # Freeze by returning via Counter (immutable use contract enforced by callers).
    return result


def cosine_similarity(vec_a: Counter[str], vec_b: Counter[str]) -> float:
    if not vec_a or not vec_b:
        return 0.0

    dot = 0.0
    for token, value in vec_a.items():
        dot += value * vec_b.get(token, 0)

    norm_a = sqrt(sum(value * value for value in vec_a.values()))
    norm_b = sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def find_duplicates(
    title: str,
    existing_titles: dict[int, str],
    *,
    threshold: float = 0.92,
) -> list[tuple[int, float]]:
    """Return (paper_id, similarity) pairs where title similarity >= threshold."""
    target_vec = build_vector(title.lower())
    if not target_vec:
        return []

    results: list[tuple[int, float]] = []
    for paper_id, existing_title in existing_titles.items():
        other_vec = build_vector(existing_title.lower())
        sim = cosine_similarity(target_vec, other_vec)
        if sim >= threshold:
            results.append((paper_id, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def top_related_papers_embedding(paper_id: int, top_k: int = 3) -> list[int]:
    """Use FAISS embeddings to find related papers. Returns paper IDs or empty list."""
    try:
        from app.services.embeddings import get_embedding_service

        service = get_embedding_service()
        if service.index_count() == 0:
            return []
        results = service.search_by_id(paper_id, top_k=top_k)
        return [pid for pid, _score in results]
    except Exception:
        LOGGER.debug("Embedding-based related papers unavailable", exc_info=True)
        return []


def top_related_papers(
    paper_id: int,
    vectors_by_id: dict[int, Counter[str]],
    *,
    top_k: int = 3,
    min_similarity: float = 0.18,
) -> list[int]:
    # Try embedding-based similarity first (higher quality)
    embedding_results = top_related_papers_embedding(paper_id, top_k=top_k)
    if embedding_results:
        return embedding_results

    # Fall back to TF-IDF
    target = vectors_by_id.get(paper_id)
    if not target:
        return []

    scored: list[tuple[float, int]] = []
    for other_id, other_vector in vectors_by_id.items():
        if other_id == paper_id:
            continue

        similarity = cosine_similarity(target, other_vector)
        if similarity >= min_similarity:
            scored.append((similarity, other_id))

    scored.sort(reverse=True)
    return [other_id for _, other_id in scored[:top_k]]
