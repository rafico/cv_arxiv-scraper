"""Related-paper recommendation helpers."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from math import sqrt

from app.services.text import STOP_WORDS, tokenize


@lru_cache(maxsize=512)
def build_vector(text: str) -> Counter[str]:
    return Counter(token for token in tokenize(text) if token not in STOP_WORDS)


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


def top_related_papers(
    paper_id: int,
    vectors_by_id: dict[int, Counter[str]],
    *,
    top_k: int = 3,
    min_similarity: float = 0.18,
) -> list[int]:
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
