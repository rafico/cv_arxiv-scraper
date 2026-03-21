"""Scoring and ranking helpers."""

from __future__ import annotations

from datetime import date
from math import exp

from app.services.preferences import get_preferences
from app.services.text import utc_today

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
    "save": 10,
    "skip": -9,
    "ignore": -3,
    "skimmed": 2,
    "priority": 20,
    "shared": 15,
}


def resolve_ranking_preferences(config: dict | None = None) -> dict[str, float]:
    preferences = get_preferences(config)
    return {
        "Author": float(preferences["ranking"]["author_weight"]),
        "Affiliation": float(preferences["ranking"]["affiliation_weight"]),
        "Title": float(preferences["ranking"]["title_weight"]),
        "ai_weight": float(preferences["ranking"]["ai_weight"]),
        "citation_weight": float(preferences.get("ranking", {}).get("citation_weight", 0.5)),
        "half_life_days": float(preferences["ranking"]["freshness_half_life_days"]),
    }


def recency_multiplier(
    publication_dt: date | None,
    today: date | None = None,
    *,
    half_life_days: float = HALF_LIFE_DAYS,
) -> float:
    """Recency decay with a gentle half-life to keep fresh papers near the top."""
    if publication_dt is None:
        return 0.72

    today = today or utc_today()
    age_days = max(0, (today - publication_dt).days)
    effective_half_life = max(0.5, float(half_life_days))
    return exp(-age_days / effective_half_life)


def compute_paper_score(
    *,
    match_types: list[str],
    matched_terms_count: int,
    publication_dt: date | None,
    resource_count: int,
    llm_relevance_score: float | None = None,
    citation_count: int | None = None,
    config: dict | None = None,
) -> float:
    preferences = resolve_ranking_preferences(config)
    match_score = sum(preferences.get(match_type, MATCH_TYPE_WEIGHTS.get(match_type, 0.0)) for match_type in match_types)
    term_score = matched_terms_count * TERM_MATCH_WEIGHT
    resource_score = min(resource_count, 4) * RESOURCE_SIGNAL_WEIGHT
    llm_bonus = (
        (llm_relevance_score / 10.0) * preferences["ai_weight"]
        if llm_relevance_score is not None
        else 0.0
    )
    citation_bonus = 0.0
    if citation_count and citation_count > 0:
        import math
        citation_bonus = math.log1p(citation_count) * preferences["citation_weight"]

    recency = recency_multiplier(publication_dt, half_life_days=preferences["half_life_days"])

    return round((match_score + term_score + resource_score + llm_bonus + citation_bonus) * recency, 3)


def explain_score(
    *,
    match_types: list[str],
    matched_terms_count: int,
    publication_dt: date | None,
    resource_count: int,
    llm_relevance_score: float | None = None,
    citation_count: int | None = None,
    feedback_score: int = 0,
    config: dict | None = None,
) -> dict[str, float]:
    preferences = resolve_ranking_preferences(config)
    match_score = sum(preferences.get(match_type, 0.0) for match_type in match_types)
    term_score = matched_terms_count * TERM_MATCH_WEIGHT
    resource_score = min(resource_count, 4) * RESOURCE_SIGNAL_WEIGHT
    ai_bonus = (
        (llm_relevance_score / 10.0) * preferences["ai_weight"]
        if llm_relevance_score is not None
        else 0.0
    )
    citation_bonus = 0.0
    if citation_count and citation_count > 0:
        import math
        citation_bonus = math.log1p(citation_count) * preferences["citation_weight"]

    recency = recency_multiplier(publication_dt, half_life_days=preferences["half_life_days"])
    base_score = round((match_score + term_score + resource_score + ai_bonus + citation_bonus) * recency, 3)
    feedback_bonus = round(feedback_score * FEEDBACK_BOOST, 3)
    return {
        "match_score": round(match_score, 3),
        "term_score": round(term_score, 3),
        "resource_score": round(resource_score, 3),
        "ai_bonus": round(ai_bonus, 3),
        "citation_bonus": round(citation_bonus, 3),
        "recency_multiplier": round(recency, 3),
        "base_score": base_score,
        "feedback_bonus": feedback_bonus,
        "rank_score": round(base_score + feedback_bonus, 3),
    }


def recompute_all_paper_scores(app, *, batch_size: int = 500) -> int:
    from app.models import Paper, db

    updated = 0
    with app.app_context():
        config = app.config["SCRAPER_CONFIG"]
        offset = 0
        while True:
            papers = Paper.query.order_by(Paper.id).offset(offset).limit(batch_size).all()
            if not papers:
                break
            for paper in papers:
                paper.paper_score = compute_paper_score(
                    match_types=[part.strip() for part in (paper.match_type or "").split("+") if part.strip()],
                    matched_terms_count=len(paper.matched_terms_list),
                    publication_dt=paper.publication_dt,
                    resource_count=len(paper.resource_links_list),
                    llm_relevance_score=paper.llm_relevance_score,
                    citation_count=paper.citation_count,
                    config=config,
                )
                updated += 1
            db.session.commit()
            offset += batch_size
    return updated


def compute_feedback_delta(action: str) -> int:
    return FEEDBACK_WEIGHTS.get(action, 0)


def combined_rank_score(paper_score: float, feedback_score: int) -> float:
    return round(paper_score + (feedback_score * FEEDBACK_BOOST), 3)


def generate_ranking_explanation(paper, config: dict | None = None) -> list[str]:
    """Generate human-readable explanation strings for why a paper was ranked."""
    explanations = []

    # Match type explanations
    match_types = [part.strip() for part in (paper.match_type or "").split("+") if part.strip()]
    matched_terms = paper.matched_terms_list[:3]
    for mt in match_types:
        if mt == "Author":
            if matched_terms:
                explanations.append(f"Matched author: {matched_terms[0]}")
            else:
                explanations.append("Matched author in your watchlist")
        elif mt == "Affiliation":
            if matched_terms:
                explanations.append(f"From tracked institution: {matched_terms[0]}")
            else:
                explanations.append("From a tracked institution")
        elif mt == "Title":
            if matched_terms:
                explanations.append(f"Title matches: {', '.join(matched_terms)}")
            else:
                explanations.append("Title matches your interests")

    # Citation explanation
    if paper.citation_count and paper.citation_count > 10:
        explanations.append(f"Highly cited ({paper.citation_count} citations)")

    # Recency explanation
    breakdown = explain_score(
        match_types=match_types,
        matched_terms_count=len(paper.matched_terms_list),
        publication_dt=paper.publication_dt,
        resource_count=len(paper.resource_links_list),
        llm_relevance_score=paper.llm_relevance_score,
        citation_count=paper.citation_count,
        config=config,
    )
    if breakdown["recency_multiplier"] > 0.9:
        explanations.append("Published very recently")

    # AI relevance explanation
    if paper.llm_relevance_score and paper.llm_relevance_score >= 7:
        explanations.append(f"AI rated highly relevant ({paper.llm_relevance_score:.0f}/10)")

    # Resource availability
    if paper.resource_links_list:
        explanations.append("Code or dataset available")

    # Similar to saved paper (embedding-based)
    try:
        from app.services.embeddings import get_embedding_service
        from app.models import Paper, PaperFeedback, db

        service = get_embedding_service()
        if service.has_paper(paper.id) and service.index_count() > 0:
            saved_ids = {
                row[0]
                for row in db.session.query(PaperFeedback.paper_id)
                .filter(PaperFeedback.action.in_(["save", "priority"]))
                .all()
            }
            if saved_ids:
                similar = service.search_by_id(paper.id, top_k=5)
                for pid, score in similar:
                    if pid in saved_ids and score > 0.5:
                        saved_paper = db.session.get(Paper, pid)
                        if saved_paper:
                            title = saved_paper.title[:50]
                            if len(saved_paper.title) > 50:
                                title += "..."
                            explanations.append(f"Similar to saved: \"{title}\"")
                            break
    except Exception:
        pass

    return explanations
