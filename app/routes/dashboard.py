from __future__ import annotations

from datetime import timedelta

from flask import Blueprint, current_app, render_template, request
from flask_sqlalchemy.query import Query

from app.constants import ARXIV_CATEGORY_NAMES, DASHBOARD_PER_PAGE
from app.csrf import get_or_create_csrf_token
from app.enums import FeedbackAction, SortOption
from app.models import Collection, DigestRun, Paper, PaperCollection, PaperFeedback, SavedSearch, ScrapeRun, db
from app.services.feedback import get_feedback_snapshot
from app.services.preferences import first_author_name, get_preferences
from app.services.ranking import FEEDBACK_BOOST, combined_rank_score, explain_score, generate_ranking_explanation
from app.services.related import build_vector, top_related_papers
from app.services.text import now_utc

dashboard_bp = Blueprint("dashboard", __name__)

TIMEFRAME_DAYS = {
    "daily": 1,
    "weekly": 7,
    "monthly": 30,
    "all": None,
}

VIEW_OPTIONS = {"inbox", "saved"}
SORT_OPTIONS = {option.value for option in SortOption}
RESOURCE_FILTER_OPTIONS = {"all", "available", "missing"}


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

    cutoff_dt = now_utc() - timedelta(days=days)
    cutoff_date = cutoff_dt.date()
    return query.filter(
        db.or_(
            Paper.publication_dt >= cutoff_date,
            db.and_(Paper.publication_dt.is_(None), Paper.scraped_at >= cutoff_dt),
        )
    )


def _interest_counts(config: dict) -> dict[str, int]:
    whitelists = config.get("whitelists", {})
    counts = {
        "authors": len(whitelists.get("authors", [])),
        "affiliations": len(whitelists.get("affiliations", [])),
        "titles": len(whitelists.get("titles", [])),
    }
    counts["total"] = counts["authors"] + counts["affiliations"] + counts["titles"]
    return counts


def _escape_like_term(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _apply_muted_filters(query: Query, config: dict, *, active: bool) -> Query:
    if not active:
        return query

    preferences = get_preferences(config)
    muted = preferences["muted"]
    for author in muted["authors"]:
        escaped = f"%{_escape_like_term(author)}%"
        query = query.filter(~Paper.authors.ilike(escaped, escape="\\"))
    for topic in muted["topics"]:
        escaped = f"%{_escape_like_term(topic)}%"
        query = query.filter(~db.cast(Paper.topic_tags, db.Text).ilike(escaped, escape="\\"))
    for affiliation in muted["affiliations"]:
        escaped = f"%{_escape_like_term(affiliation)}%"
        query = query.filter(~db.cast(Paper.matched_terms, db.Text).ilike(escaped, escape="\\"))
    return query


def _apply_category_filter(query: Query, category: str | None) -> Query:
    if not category:
        return query
    escaped = f"%{_escape_like_term(category)}%"
    return query.filter(~db.cast(Paper.categories, db.Text).is_(None)).filter(
        db.cast(Paper.categories, db.Text).ilike(escaped, escape="\\")
    )


def _apply_resource_filter(query: Query, resource_filter: str) -> Query:
    resources_expr = db.cast(Paper.resource_links, db.Text)
    if resource_filter == "available":
        return query.filter(resources_expr != "[]")
    if resource_filter == "missing":
        return query.filter(db.or_(resources_expr == "[]", resources_expr.is_(None)))
    return query


def _build_filter_options(query: Query) -> dict:
    base = query.order_by(None)

    # Resource counts via SQL to avoid loading all rows.
    resources_expr = db.cast(Paper.resource_links, db.Text)
    resources_available = base.filter(resources_expr != "[]").count()
    total = base.count()
    resources_missing = total - resources_available

    # Category counts still need Python because SQLite has no native JSON unnest.
    category_counts: dict[str, int] = {}
    for (categories,) in base.with_entities(Paper.categories).yield_per(500):
        for category in categories or []:
            category_counts[category] = category_counts.get(category, 0) + 1

    categories = [
        {"label": label, "name": ARXIV_CATEGORY_NAMES.get(label, label), "count": count}
        for label, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0].lower()))
    ]
    return {
        "categories": categories,
        "resources": {
            "available": resources_available,
            "missing": resources_missing,
        },
    }


def _build_onboarding_steps(config: dict, *, saved_count: int, has_successful_scrape: bool) -> list[dict]:
    interest_total = _interest_counts(config)["total"]
    return [
        {
            "label": "Add interests",
            "description": "Track the authors, labs, and topics you care about.",
            "complete": interest_total > 0,
            "href": "/settings?section=interests",
        },
        {
            "label": "Run a scrape",
            "description": "Build your research inbox from the latest arXiv feed.",
            "complete": has_successful_scrape,
            "href": "#run-scrape",
        },
        {
            "label": "Save or skip papers",
            "description": "Save what matters, skip the rest. This trains your ranking.",
            "complete": saved_count > 0 or PaperFeedback.query.count() > 0,
            "href": "/",
        },
    ]


def _build_dashboard_overview(config: dict) -> dict:
    latest_scrape = ScrapeRun.query.order_by(ScrapeRun.started_at.desc()).first()
    latest_digest = DigestRun.query.order_by(DigestRun.started_at.desc()).first()
    has_successful_scrape = db.session.query(ScrapeRun.id).filter(ScrapeRun.status == "success").first() is not None

    latest_scrape_view = None
    if latest_scrape is not None:
        timestamp = latest_scrape.finished_at or latest_scrape.started_at
        latest_scrape_view = {
            "status": latest_scrape.status,
            "status_label": latest_scrape.status.replace("_", " ").title(),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M UTC"),
            "forced": bool(latest_scrape.forced),
        }

    latest_digest_view = None
    if latest_digest is not None:
        timestamp = latest_digest.finished_at or latest_digest.started_at
        latest_digest_view = {
            "status": latest_digest.status,
            "status_label": latest_digest.status.replace("_", " ").title(),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M UTC"),
            "papers_count": int(latest_digest.papers_count or 0),
            "preview_only": bool(latest_digest.preview_only),
            "recipient": latest_digest.recipient,
        }

    saved_count = PaperFeedback.query.filter_by(action=FeedbackAction.SAVE.value).count()

    return {
        "saved_count": saved_count,
        "interest_counts": _interest_counts(config),
        "latest_scrape": latest_scrape_view,
        "latest_digest": latest_digest_view,
        "has_successful_scrape": has_successful_scrape,
        "scrape_history": ScrapeRun.query.order_by(ScrapeRun.started_at.desc()).limit(5).all(),
        "digest_history": DigestRun.query.order_by(DigestRun.started_at.desc()).limit(5).all(),
        "onboarding_steps": _build_onboarding_steps(
            config,
            saved_count=saved_count,
            has_successful_scrape=has_successful_scrape,
        ),
    }


def _enrich_cards_with_feedback_and_related(papers: list[Paper], candidate_pool: list[Paper], config: dict) -> None:
    paper_ids = [paper.id for paper in papers]
    feedback_snapshot = get_feedback_snapshot(paper_ids)
    preferences = get_preferences(config)
    followed_authors = set(config.get("whitelists", {}).get("authors", []))
    muted_topics = set(preferences["muted"]["topics"])

    vectors_by_id = {
        paper.id: build_vector(
            " ".join(
                [
                    paper.title or "",
                    paper.summary_text or "",
                    paper.abstract_text or "",
                    " ".join(paper.topic_tags or []),
                ]
            )
        )
        for paper in candidate_pool
    }
    candidate_by_id = {paper.id: paper for paper in candidate_pool}

    for paper in papers:
        feedback = feedback_snapshot.get(
            paper.id,
            {"counts": {a.value: 0 for a in FeedbackAction}, "active_actions": []},
        )
        paper.feedback_counts = feedback["counts"]  # type: ignore[attr-defined]
        paper.active_actions = feedback["active_actions"]  # type: ignore[attr-defined]
        paper.rank_score_value = combined_rank_score(float(paper.paper_score or 0.0), int(paper.feedback_score or 0))  # type: ignore[attr-defined]
        paper.score_breakdown = explain_score(  # type: ignore[attr-defined]
            match_types=[part.strip() for part in (paper.match_type or "").split("+") if part.strip()],
            matched_terms_count=len(paper.matched_terms_list),
            publication_dt=paper.publication_dt,
            resource_count=len(paper.resource_links_list),
            llm_relevance_score=paper.llm_relevance_score,
            feedback_score=int(paper.feedback_score or 0),
            config=config,
        )
        primary_author = first_author_name(paper.authors)
        primary_topic = next((topic for topic in paper.topic_tags_list if topic), "")
        paper.follow_recommendation = {  # type: ignore[attr-defined]
            "label": primary_author,
            "available": bool(primary_author) and primary_author not in followed_authors,
        }
        paper.mute_recommendation = {  # type: ignore[attr-defined]
            "label": primary_topic,
            "available": bool(primary_topic) and primary_topic not in muted_topics,
        }

        related_ids = top_related_papers(paper.id, vectors_by_id, top_k=3)
        paper.related_papers = [
            candidate_by_id[related_id] for related_id in related_ids if related_id in candidate_by_id
        ]  # type: ignore[attr-defined]

        paper.ranking_explanations = generate_ranking_explanation(paper, config=config)  # type: ignore[attr-defined]


@dashboard_bp.route("/")
def index():
    config = current_app.config["SCRAPER_CONFIG"]
    view = request.args.get("view", "inbox")
    if view not in VIEW_OPTIONS:
        view = "inbox"

    collection_id = request.args.get("collection", type=int)

    query = Paper.query
    if collection_id:
        query = query.join(
            PaperCollection,
            db.and_(PaperCollection.paper_id == Paper.id, PaperCollection.collection_id == collection_id),
        )
    elif view == "saved":
        query = query.join(
            PaperFeedback,
            db.and_(PaperFeedback.paper_id == Paper.id, PaperFeedback.action == FeedbackAction.SAVE.value),
        )
    query = _apply_muted_filters(query, config, active=view != "saved" and not collection_id)

    include_hidden = request.args.get("include_hidden") == "1"
    if not include_hidden:
        query = query.filter(Paper.is_hidden.is_(False))

    default_timeframe = "all" if view == "saved" else "daily"
    timeframe = request.args.get("timeframe", default_timeframe)
    if timeframe not in TIMEFRAME_DAYS:
        timeframe = default_timeframe
    query = _apply_timeframe(query, timeframe)

    match_type = request.args.get("match_type")
    if match_type:
        query = query.filter(Paper.match_type.contains(match_type))

    q = request.args.get("q", "").strip()
    search_mode = request.args.get("search_mode", "hybrid").strip()
    hybrid_search_used = False
    if q:
        # Try hybrid/semantic search when available
        if search_mode in ("hybrid", "semantic"):
            try:
                from app.services.search import search_hybrid, search_semantic

                if search_mode == "semantic":
                    raw = search_semantic(q, top_k=100)
                    hybrid_ids = [pid for pid, _ in raw]
                else:
                    hybrid_results = search_hybrid(q, top_k=100)
                    hybrid_ids = [r["paper_id"] for r in hybrid_results]

                if hybrid_ids:
                    query = query.filter(Paper.id.in_(hybrid_ids))
                    hybrid_search_used = True
            except Exception:
                pass

        if not hybrid_search_used:
            escaped_q = _escape_like_term(q)
            search = f"%{escaped_q}%"
            query = query.filter(
                db.or_(
                    Paper.title.ilike(search, escape="\\"),
                    Paper.authors.ilike(search, escape="\\"),
                    Paper.abstract_text.ilike(search, escape="\\"),
                    db.cast(Paper.matched_terms, db.Text).ilike(search, escape="\\"),
                    Paper.summary_text.ilike(search, escape="\\"),
                    db.cast(Paper.topic_tags, db.Text).ilike(search, escape="\\"),
                    db.cast(Paper.user_tags, db.Text).ilike(search, escape="\\"),
                )
            )

    reading_status = request.args.get("reading_status", "").strip()
    if reading_status:
        if reading_status == "unread":
            query = query.filter(Paper.reading_status.is_(None))
        else:
            query = query.filter(Paper.reading_status == reading_status)

    author_filter = request.args.get("author", "").strip()
    if author_filter:
        escaped_author = f"%{_escape_like_term(author_filter)}%"
        query = query.filter(Paper.authors.ilike(escaped_author, escape="\\"))

    category = request.args.get("category", "").strip()
    resource_filter = request.args.get("resource_filter", "all").strip()
    if resource_filter not in RESOURCE_FILTER_OPTIONS:
        resource_filter = "all"

    filter_options = _build_filter_options(query)
    query = _apply_category_filter(query, category or None)
    query = _apply_resource_filter(query, resource_filter)

    default_sort = "saved" if view == "saved" else "trending"
    sort = request.args.get("sort", default_sort)
    valid_sorts = (
        {"saved", "newest", "citations"} if view == "saved" else {"trending", "newest", "recommended", "citations"}
    )
    if sort not in valid_sorts or sort not in SORT_OPTIONS:
        sort = default_sort

    if sort == "saved" and view == "saved":
        query = query.order_by(
            PaperFeedback.created_at.desc(),
            Paper.publication_dt.desc(),
            Paper.scraped_at.desc(),
        )
    elif sort == "newest":
        query = query.order_by(Paper.publication_dt.desc(), Paper.scraped_at.desc())
    elif sort == SortOption.CITATIONS.value:
        query = query.order_by(
            db.func.coalesce(Paper.citation_count, 0).desc(),
            Paper.publication_dt.desc(),
            Paper.scraped_at.desc(),
        )
    elif sort == "recommended":
        query = query.order_by(
            db.func.coalesce(Paper.recommendation_score, 0.0).desc(),
            Paper.publication_dt.desc(),
            Paper.scraped_at.desc(),
        )
    else:
        query = query.order_by(
            (
                db.func.coalesce(Paper.paper_score, 0.0) + db.func.coalesce(Paper.feedback_score, 0) * FEEDBACK_BOOST
            ).desc(),
            Paper.publication_dt.desc(),
            Paper.scraped_at.desc(),
        )

    page = _parse_page(request.args.get("page"))
    pagination = query.paginate(page=page, per_page=DASHBOARD_PER_PAGE, error_out=False)
    papers = pagination.items

    type_counts_row = (
        query.order_by(None)
        .with_entities(
            db.func.sum(db.case((Paper.match_type.contains("Author"), 1), else_=0)).label("author_count"),
            db.func.sum(db.case((Paper.match_type.contains("Affiliation"), 1), else_=0)).label("affiliation_count"),
            db.func.sum(db.case((Paper.match_type.contains("Title"), 1), else_=0)).label("title_count"),
        )
        .first()
    )
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
    _enrich_cards_with_feedback_and_related(papers, candidate_pool, config)

    return render_template(
        "dashboard.html",
        papers=papers,
        pagination=pagination,
        type_counts=type_counts,
        current_filters={
            "view": view,
            "match_type": match_type,
            "q": q,
            "search_mode": search_mode,
            "timeframe": timeframe,
            "sort": sort,
            "include_hidden": include_hidden,
            "category": category,
            "resource_filter": resource_filter,
            "reading_status": reading_status,
            "author": author_filter,
            "collection": collection_id,
        },
        filter_options=filter_options,
        dashboard_overview=_build_dashboard_overview(config),
        collections=Collection.query.order_by(Collection.name).all(),
        saved_searches=SavedSearch.query.order_by(SavedSearch.created_at.desc()).all(),
        csrf_token=get_or_create_csrf_token(),
    )
