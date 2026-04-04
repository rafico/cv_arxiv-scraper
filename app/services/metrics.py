"""Recommendation quality metrics computed from local feedback history."""

from __future__ import annotations

from datetime import date

from app.enums import FeedbackAction
from app.models import Paper, PaperFeedback, RecommendationMetric, db
from app.services.matching import check_author_match
from app.services.ranking import (
    build_ranking_config_snapshot,
    compute_paper_score,
    get_active_ranking_config,
)

POSITIVE_FEEDBACK_ACTIONS = {
    FeedbackAction.SAVE.value,
    FeedbackAction.PRIORITY.value,
}
OPEN_FEEDBACK_ACTIONS = {
    FeedbackAction.SAVE.value,
    FeedbackAction.PRIORITY.value,
    FeedbackAction.SKIMMED.value,
    FeedbackAction.SHARED.value,
}


def _resolve_scraper_config(config: dict | None = None) -> dict | None:
    if config is not None:
        return config

    try:
        from flask import current_app, has_app_context

        if has_app_context():
            return current_app.config.get("SCRAPER_CONFIG")
    except Exception:
        return None
    return None


def _match_types_for_paper(paper: Paper) -> list[str]:
    return [part.strip() for part in (paper.match_type or "").split("+") if part.strip()]


def _rank_papers(
    papers: list[Paper],
    *,
    config: dict | None = None,
    ranking_config=None,
) -> list[tuple[Paper, float]]:
    scored = []
    for paper in papers:
        scored.append(
            (
                paper,
                compute_paper_score(
                    match_types=_match_types_for_paper(paper),
                    matched_terms_count=len(paper.matched_terms_list),
                    publication_dt=paper.publication_dt,
                    resource_count=len(paper.resource_links_list),
                    llm_relevance_score=paper.llm_relevance_score,
                    citation_count=paper.citation_count,
                    config=config,
                    ranking_config=ranking_config,
                ),
            )
        )

    scored.sort(
        key=lambda item: (
            -item[1],
            -(item[0].paper_score or 0.0),
            -(item[0].publication_dt.toordinal() if item[0].publication_dt else date.min.toordinal()),
            -item[0].id,
        ),
    )
    return scored


def _positive_feedback_paper_ids() -> set[int]:
    rows = (
        db.session.query(PaperFeedback.paper_id)
        .filter(PaperFeedback.action.in_(sorted(POSITIVE_FEEDBACK_ACTIONS)))
        .distinct()
        .all()
    )
    return {int(row[0]) for row in rows}


def compute_precision_at_k(
    top_k: int = 10,
    *,
    papers: list[Paper] | None = None,
    positive_paper_ids: set[int] | None = None,
    config: dict | None = None,
    ranking_config=None,
) -> float:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    papers = papers if papers is not None else Paper.query.filter(Paper.is_hidden.is_(False)).all()
    positive_paper_ids = positive_paper_ids if positive_paper_ids is not None else _positive_feedback_paper_ids()
    ranked = _rank_papers(papers, config=config, ranking_config=ranking_config)
    top_ranked = ranked[:top_k]
    if not top_ranked:
        return 0.0
    positives = sum(1 for paper, _score in top_ranked if paper.id in positive_paper_ids)
    return round(positives / len(top_ranked), 4)


def compute_author_follow_hit_rate(
    *,
    tracked_authors: list[str] | None = None,
    papers: list[Paper] | None = None,
    positive_paper_ids: set[int] | None = None,
    config: dict | None = None,
) -> float:
    resolved_config = _resolve_scraper_config(config) or {}
    tracked_authors = (
        tracked_authors if tracked_authors is not None else resolved_config.get("whitelists", {}).get("authors", [])
    )
    tracked_authors = [author for author in tracked_authors if author]
    if not tracked_authors:
        return 0.0

    papers = papers if papers is not None else Paper.query.filter(Paper.is_hidden.is_(False)).all()
    positive_paper_ids = positive_paper_ids if positive_paper_ids is not None else _positive_feedback_paper_ids()

    tracked_papers = [
        paper
        for paper in papers
        if check_author_match([name.strip() for name in paper.authors.split(",") if name.strip()], tracked_authors)
    ]
    if not tracked_papers:
        return 0.0

    saved_count = sum(1 for paper in tracked_papers if paper.id in positive_paper_ids)
    return round(saved_count / len(tracked_papers), 4)


def compute_mean_time_to_first_open_hours(
    *,
    papers: list[Paper] | None = None,
) -> float:
    papers = papers if papers is not None else Paper.query.all()
    if not papers:
        return 0.0

    feedback_rows = (
        PaperFeedback.query.filter(PaperFeedback.action.in_(sorted(OPEN_FEEDBACK_ACTIONS)))
        .order_by(PaperFeedback.created_at.asc(), PaperFeedback.id.asc())
        .all()
    )

    first_open_by_paper: dict[int, object] = {}
    for row in feedback_rows:
        first_open_by_paper.setdefault(row.paper_id, row.created_at)

    deltas_hours: list[float] = []
    for paper in papers:
        opened_at = first_open_by_paper.get(paper.id)
        if opened_at is None or paper.scraped_at is None:
            continue
        delta_hours = max((opened_at - paper.scraped_at).total_seconds() / 3600.0, 0.0)
        deltas_hours.append(delta_hours)

    if not deltas_hours:
        return 0.0
    return round(sum(deltas_hours) / len(deltas_hours), 4)


def measure_recommendation_quality(
    *,
    config: dict | None = None,
    ranking_configs: list | None = None,
    precision_ks: tuple[int, ...] = (10,),
    tracked_authors: list[str] | None = None,
    persist: bool = True,
) -> list[dict]:
    resolved_config = _resolve_scraper_config(config)
    visible_papers = Paper.query.filter(Paper.is_hidden.is_(False)).all()
    all_papers = Paper.query.all()
    positive_paper_ids = _positive_feedback_paper_ids()

    configs_to_measure = ranking_configs
    if configs_to_measure is None:
        active_config = get_active_ranking_config()
        configs_to_measure = [active_config] if active_config is not None else [None]
    elif not configs_to_measure:
        configs_to_measure = [None]

    measurements: list[dict] = []
    for ranking_config in configs_to_measure:
        snapshot = build_ranking_config_snapshot(resolved_config, ranking_config=ranking_config)
        metrics: dict[str, float] = {}
        for top_k in precision_ks:
            metric_name = f"precision_at_{int(top_k)}"
            metrics[metric_name] = compute_precision_at_k(
                int(top_k),
                papers=visible_papers,
                positive_paper_ids=positive_paper_ids,
                config=resolved_config,
                ranking_config=ranking_config,
            )

        metrics["author_follow_hit_rate"] = compute_author_follow_hit_rate(
            tracked_authors=tracked_authors,
            papers=visible_papers,
            positive_paper_ids=positive_paper_ids,
            config=resolved_config,
        )
        metrics["mean_time_to_first_open_hours"] = compute_mean_time_to_first_open_hours(papers=all_papers)

        if persist:
            for metric_name, metric_value in metrics.items():
                db.session.add(
                    RecommendationMetric(
                        metric_name=metric_name,
                        metric_value=float(metric_value),
                        config_snapshot=snapshot,
                    )
                )

        measurements.append({"config_snapshot": snapshot, "metrics": metrics})

    if persist:
        db.session.commit()
    return measurements


def compare_metric_outcomes_across_configs(
    ranking_configs: list,
    *,
    config: dict | None = None,
    precision_ks: tuple[int, ...] = (10,),
    tracked_authors: list[str] | None = None,
    persist: bool = True,
) -> list[dict]:
    return measure_recommendation_quality(
        config=config,
        ranking_configs=ranking_configs,
        precision_ks=precision_ks,
        tracked_authors=tracked_authors,
        persist=persist,
    )
