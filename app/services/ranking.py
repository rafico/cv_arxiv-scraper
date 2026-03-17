"""Scoring and ranking helpers."""

from __future__ import annotations

from datetime import date
from math import exp

MATCH_TYPE_WEIGHTS = {
    "Author": 44.0,
    "Affiliation": 26.0,
    "Title": 14.0,
}

TERM_MATCH_WEIGHT = 3.0
RESOURCE_SIGNAL_WEIGHT = 1.5
LLM_RELEVANCE_WEIGHT = 5.0
HALF_LIFE_DAYS = 14.0

FEEDBACK_BOOST = 1.25

FEEDBACK_WEIGHTS = {
    "upvote": 5,
    "save": 7,
    "skip": -9,
}


def recency_multiplier(publication_dt: date | None, today: date | None = None) -> float:
    """Recency decay with a gentle half-life to keep fresh papers near the top."""
    if publication_dt is None:
        return 0.72

    today = today or date.today()
    age_days = max(0, (today - publication_dt).days)
    return exp(-age_days / HALF_LIFE_DAYS)


def compute_paper_score(
    *,
    match_types: list[str],
    matched_terms_count: int,
    publication_dt: date | None,
    resource_count: int,
    llm_relevance_score: float | None = None,
) -> float:
    match_score = sum(MATCH_TYPE_WEIGHTS.get(match_type, 0.0) for match_type in match_types)
    term_score = matched_terms_count * TERM_MATCH_WEIGHT
    resource_score = min(resource_count, 4) * RESOURCE_SIGNAL_WEIGHT
    llm_bonus = (
        (llm_relevance_score / 10.0) * LLM_RELEVANCE_WEIGHT
        if llm_relevance_score is not None
        else 0.0
    )
    recency = recency_multiplier(publication_dt)

    return round((match_score + term_score + resource_score + llm_bonus) * recency, 3)


def compute_feedback_delta(action: str) -> int:
    return FEEDBACK_WEIGHTS.get(action, 0)


def combined_rank_score(paper_score: float, feedback_score: int) -> float:
    return round(paper_score + (feedback_score * FEEDBACK_BOOST), 3)
