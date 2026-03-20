"""LLM-powered smart paper recommendations based on user feedback history."""

from __future__ import annotations

import logging

from app.models import Paper, PaperFeedback, db

LOGGER = logging.getLogger(__name__)


def build_preference_profile(limit: int = 50) -> str:
    """Build a text profile from the user's saved/upvoted papers."""
    positive_paper_ids = (
        db.session.query(PaperFeedback.paper_id)
        .filter(PaperFeedback.action.in_(["save", "upvote"]))
        .order_by(PaperFeedback.created_at.desc())
        .limit(limit)
        .subquery()
    )
    papers = Paper.query.filter(Paper.id.in_(positive_paper_ids)).all()
    if not papers:
        return ""

    parts = []
    for paper in papers:
        parts.append(f"- {paper.title}")
        if paper.topic_tags:
            parts.append(f"  Topics: {', '.join(paper.topic_tags[:5])}")
    return "Papers the user has saved or upvoted:\n" + "\n".join(parts)


def score_papers_with_llm(
    papers: list[Paper],
    llm_client,
    preference_profile: str,
) -> dict[int, float]:
    """Score papers against user preferences using the LLM. Returns {paper_id: score}."""
    if not preference_profile or not papers or not llm_client:
        return {}

    scores: dict[int, float] = {}
    for paper in papers:
        try:
            score = llm_client.rate_relevance(
                paper.title,
                paper.abstract_text or "",
                preference_profile,
            )
            if score is not None:
                scores[paper.id] = float(score)
        except Exception:
            LOGGER.debug("Failed to score paper %s for recommendations", paper.id)

    return scores


def update_recommendation_scores(app) -> int:
    """Recompute recommendation scores for recent unscored papers."""
    from pathlib import Path

    from app.services.scrape_engine import _create_llm_client

    llm_client, _ = _create_llm_client(app)
    if not llm_client:
        return 0

    with app.app_context():
        profile = build_preference_profile()
        if not profile:
            return 0

        unscored = (
            Paper.query
            .filter(Paper.recommendation_score.is_(None))
            .order_by(Paper.scraped_at.desc())
            .limit(50)
            .all()
        )
        if not unscored:
            return 0

        scores = score_papers_with_llm(unscored, llm_client, profile)
        for paper in unscored:
            paper.recommendation_score = scores.get(paper.id)
        db.session.commit()
        return len(scores)
