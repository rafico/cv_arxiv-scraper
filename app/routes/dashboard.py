from __future__ import annotations

from datetime import datetime, timedelta, timezone

from flask import Blueprint, render_template, request
from flask_sqlalchemy.query import Query

from app.models import Paper, db
from app.services.feedback import get_feedback_snapshot
from app.services.related import build_vector, top_related_papers
from app.constants import DASHBOARD_PER_PAGE
from app.services.ranking import FEEDBACK_BOOST, combined_rank_score

dashboard_bp = Blueprint("dashboard", __name__)

TIMEFRAME_DAYS = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
    "all": None,
}

SORT_OPTIONS = {"trending", "newest"}


def _parse_page(raw_value: str | None) -> int:
    try:
        value = int(raw_value or "1")
        return value if value > 0 else 1
    except ValueError:
        return 1


def _apply_timeframe(query: Query, timeframe: str) -> Query:
    days = TIMEFRAME_DAYS.get(timeframe)
    if days is None:
        return query

    cutoff_dt = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    cutoff_date = cutoff_dt.date()
    return query.filter(
        db.or_(
            Paper.publication_dt >= cutoff_date,
            db.and_(Paper.publication_dt.is_(None), Paper.scraped_at >= cutoff_dt),
        )
    )


def _enrich_cards_with_feedback_and_related(papers: list[Paper], candidate_pool: list[Paper]) -> None:
    paper_ids = [paper.id for paper in papers]
    feedback_snapshot = get_feedback_snapshot(paper_ids)

    vectors_by_id = {
        paper.id: build_vector(
            " ".join(
                [
                    paper.title or "",
                    paper.summary_text or "",
                    paper.abstract_text or "",
                    paper.topic_tags or "",
                ]
            )
        )
        for paper in candidate_pool
    }
    candidate_by_id = {paper.id: paper for paper in candidate_pool}

    for paper in papers:
        feedback = feedback_snapshot.get(
            paper.id,
            {"counts": {"upvote": 0, "save": 0, "skip": 0}, "active_actions": set()},
        )
        paper.feedback_counts = feedback["counts"]  # type: ignore[attr-defined]
        paper.active_actions = feedback["active_actions"]  # type: ignore[attr-defined]
        paper.rank_score_value = combined_rank_score(float(paper.paper_score or 0.0), int(paper.feedback_score or 0))  # type: ignore[attr-defined]

        related_ids = top_related_papers(paper.id, vectors_by_id, top_k=3)
        paper.related_papers = [candidate_by_id[related_id] for related_id in related_ids]  # type: ignore[attr-defined]


@dashboard_bp.route("/")
def index():
    query = Paper.query
    include_hidden = request.args.get("include_hidden") == "1"
    if not include_hidden:
        query = query.filter(Paper.is_hidden.is_(False))

    timeframe = request.args.get("timeframe", "daily")
    if timeframe not in TIMEFRAME_DAYS:
        timeframe = "daily"
    query = _apply_timeframe(query, timeframe)

    match_type = request.args.get("match_type")
    if match_type:
        query = query.filter(Paper.match_type.contains(match_type))

    q = request.args.get("q", "").strip()
    if q:
        search = f"%{q}%"
        query = query.filter(
            db.or_(
                Paper.title.ilike(search),
                Paper.authors.ilike(search),
                Paper.matched_terms.ilike(search),
                Paper.summary_text.ilike(search),
                Paper.topic_tags.ilike(search),
            )
        )

    sort = request.args.get("sort", "trending")
    if sort not in SORT_OPTIONS:
        sort = "trending"

    if sort == "newest":
        query = query.order_by(Paper.publication_dt.desc(), Paper.scraped_at.desc())
    else:
        query = query.order_by(
            (db.func.coalesce(Paper.paper_score, 0.0) + db.func.coalesce(Paper.feedback_score, 0) * FEEDBACK_BOOST).desc(),
            Paper.publication_dt.desc(),
            Paper.scraped_at.desc(),
        )

    page = _parse_page(request.args.get("page"))
    pagination = query.paginate(page=page, per_page=DASHBOARD_PER_PAGE, error_out=False)
    papers = pagination.items

    type_counts_row = query.order_by(None).with_entities(
        db.func.sum(db.case((Paper.match_type.contains("Author"), 1), else_=0)).label("author_count"),
        db.func.sum(db.case((Paper.match_type.contains("Affiliation"), 1), else_=0)).label("affiliation_count"),
        db.func.sum(db.case((Paper.match_type.contains("Title"), 1), else_=0)).label("title_count"),
    ).first()
    type_counts = {
        "Author": int(type_counts_row.author_count or 0),  # type: ignore[union-attr]
        "Affiliation": int(type_counts_row.affiliation_count or 0),  # type: ignore[union-attr]
        "Title": int(type_counts_row.title_count or 0),  # type: ignore[union-attr]
    }

    candidate_pool = (
        query.order_by(None)
        .order_by(Paper.paper_score.desc(), Paper.publication_dt.desc(), Paper.scraped_at.desc())
        .limit(250)
        .all()
    )
    _enrich_cards_with_feedback_and_related(papers, candidate_pool)

    return render_template(
        "dashboard.html",
        papers=papers,
        pagination=pagination,
        type_counts=type_counts,
        current_filters={
            "match_type": match_type,
            "q": q,
            "timeframe": timeframe,
            "sort": sort,
            "include_hidden": include_hidden,
        },
    )
