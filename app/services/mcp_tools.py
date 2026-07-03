"""MCP tool logic — the testable backend behind :mod:`app.mcp_server`.

Plain functions that expose the personalized, enriched, ranked, full-text-indexed
corpus as compact JSON-serialisable dicts. They use *only* Flask/SQLAlchemy plus
the existing services — **no ``mcp`` SDK import lives here** — so the whole layer
is unit-testable against a seeded DB without the optional MCP extra installed.

Every function runs inside the *current* Flask app context (the MCP wrapper pushes
one per call; tests run inside ``FlaskDBTestCase``'s pushed context). Read paths
are the default; the single mutation (:func:`add_to_collection`) is clearly
separated and never invoked by the read tools.

Design rules mirrored from the rest of the app:

* Degrade gracefully — an unknown id or a search backend that is unavailable
  returns a structured ``{"error": ...}``/empty payload, never an exception that
  would crash an assistant's tool call.
* Return structured data only (no HTML); the caller renders it.
* Keep dicts compact — assistants pay for every token of context.
"""

from __future__ import annotations

from typing import Any

from app.enums import SortOption
from app.models import Collection, Paper, PaperCollection, db, inbox_freshness_clause
from app.services.implementation_readiness import implementation_readiness
from app.services.preferences import first_author_name

# Bounds so an assistant can never ask for an unbounded result set (each row costs
# context tokens) or a pathological negative/zero limit.
_MAX_LIMIT = 50
_DEFAULT_LIMIT = 10
_ABSTRACT_CHARS = 600
_SUMMARY_CHARS = 1500

SEARCH_MODES = ("hybrid", "semantic", "keyword")


def _clamp_limit(limit: int | None, *, default: int = _DEFAULT_LIMIT) -> int:
    try:
        value = int(limit) if limit is not None else default
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return min(value, _MAX_LIMIT)


def _truncate(text: str | None, limit: int) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) > limit:
        return cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def _publication_str(paper: Paper) -> str | None:
    if paper.publication_dt is not None:
        return paper.publication_dt.isoformat()
    return (paper.publication_date or None) if paper.publication_date else None


def _enrichment_summary(paper: Paper) -> dict[str, Any]:
    """Compact code/citation/venue/community enrichment signals for one paper."""
    readiness = implementation_readiness(paper)
    return {
        "citation_count": paper.citation_count,
        "influential_citation_count": paper.influential_citation_count,
        "venue": paper.venue,
        "venue_year": paper.venue_year,
        "acceptance_status": paper.acceptance_status,
        "github_repo": paper.github_repo,
        "github_stars": paper.github_stars,
        "github_license": paper.github_license,
        "hf_upvotes": paper.hf_upvotes,
        "hf_comments_count": paper.hf_comments_count,
        "readiness": {
            "score": readiness.score,
            "tier": readiness.tier,
            "reasons": readiness.reasons,
        },
    }


def _paper_brief(paper: Paper) -> dict[str, Any]:
    """A compact list-item dict (search hits, ranked feeds)."""
    return {
        "id": paper.id,
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "first_author": first_author_name(paper.authors),
        "published": _publication_str(paper),
        "categories": paper.categories_list,
        "link": paper.link,
        "score": round(float(paper.paper_score or 0.0), 3),
        "citation_count": paper.citation_count,
        "github_repo": paper.github_repo,
        "abstract": _truncate(paper.abstract_text, _ABSTRACT_CHARS),
    }


def _paper_detail(paper: Paper) -> dict[str, Any]:
    """The full single-paper dict (metadata + enrichment summary)."""
    return {
        "id": paper.id,
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.authors,
        "link": paper.link,
        "pdf_link": paper.pdf_link,
        "published": _publication_str(paper),
        "categories": paper.categories_list,
        "topic_tags": paper.topic_tags_list,
        "abstract": _truncate(paper.abstract_text, _ABSTRACT_CHARS * 2),
        "summary": _truncate(paper.summary_text, _SUMMARY_CHARS),
        "score": round(float(paper.paper_score or 0.0), 3),
        "resource_links": paper.resource_links_list,
        "enrichment": _enrichment_summary(paper),
    }


def _resolve_paper(identifier: str | int) -> Paper | None:
    """Resolve a paper by numeric primary key or arXiv id.

    Accepts an int, a numeric string (tried as the PK first, then as an arXiv id),
    or an arXiv-id string (new-scheme ``2401.01234`` or old ``cs/0701001``).
    Returns ``None`` when nothing matches — callers surface a graceful error.
    """
    if isinstance(identifier, bool):  # bool is an int subclass; never a valid id
        return None
    if isinstance(identifier, int):
        return db.session.get(Paper, identifier)

    raw = str(identifier or "").strip()
    if not raw:
        return None

    # A bare integer is ambiguous: prefer the primary key, fall back to arXiv id.
    if raw.isdigit():
        by_pk = db.session.get(Paper, int(raw))
        if by_pk is not None:
            return by_pk

    # Normalise common arXiv-id spellings ("arXiv:2401.01234v2" → "2401.01234").
    arxiv_id = raw
    lowered = arxiv_id.lower()
    if lowered.startswith("arxiv:"):
        arxiv_id = arxiv_id[len("arxiv:") :]
    # Match with and without a trailing version suffix so "…v2" resolves too.
    candidates = [arxiv_id]
    if "v" in arxiv_id:
        candidates.append(arxiv_id.split("v")[0])
    for candidate in candidates:
        paper = Paper.query.filter_by(arxiv_id=candidate).first()
        if paper is not None:
            return paper
    return None


# ─────────────────────────── read tools ────────────────────────────


def search_papers(query: str, mode: str = "hybrid", limit: int = _DEFAULT_LIMIT) -> dict[str, Any]:
    """Search the corpus and return compact hits, best first.

    ``mode`` is one of ``hybrid`` (BM25 + semantic RRF, the default), ``semantic``
    (embedding similarity), or ``keyword`` (SQL substring). Unknown modes fall
    back to ``hybrid``. Empty queries and unavailable backends return no hits.
    """
    clean_query = (query or "").strip()
    normalized_mode = mode if mode in SEARCH_MODES else "hybrid"
    n = _clamp_limit(limit)
    if not clean_query:
        return {"query": clean_query, "mode": normalized_mode, "count": 0, "results": []}

    ordered_ids: list[int] = []
    if normalized_mode in ("hybrid", "semantic"):
        try:
            from app.services.search import search_hybrid, search_semantic

            if normalized_mode == "semantic":
                ordered_ids = [pid for pid, _score in search_semantic(clean_query, top_k=n)]
            else:
                ordered_ids = [row["paper_id"] for row in search_hybrid(clean_query, top_k=n)]
        except Exception:  # noqa: BLE001 — search backends are best-effort; degrade to keyword
            ordered_ids = []

    if not ordered_ids and normalized_mode != "semantic":
        ordered_ids = _keyword_ids(clean_query, n)

    papers_by_id = {p.id: p for p in Paper.query.filter(Paper.id.in_(ordered_ids)).all()} if ordered_ids else {}
    results = [_paper_brief(papers_by_id[pid]) for pid in ordered_ids if pid in papers_by_id]
    return {"query": clean_query, "mode": normalized_mode, "count": len(results), "results": results}


def _keyword_ids(query: str, limit: int) -> list[int]:
    """SQL substring fallback over the salient text columns, rank-ordered."""
    from app.services.ranking import rank_score_order_expr

    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    like = f"%{escaped}%"
    rows = (
        Paper.query.filter(Paper.is_hidden.is_(False))
        .filter(
            db.or_(
                Paper.title.ilike(like, escape="\\"),
                Paper.authors.ilike(like, escape="\\"),
                Paper.abstract_text.ilike(like, escape="\\"),
                Paper.summary_text.ilike(like, escape="\\"),
            )
        )
        .order_by(rank_score_order_expr().desc(), Paper.publication_dt.desc())
        .limit(limit)
        .with_entities(Paper.id)
        .all()
    )
    return [row[0] for row in rows]


def get_paper(arxiv_id_or_id: str | int) -> dict[str, Any]:
    """Return one paper's full metadata + citation/readiness/enrichment summary.

    Accepts a numeric primary key or an arXiv id. Unknown ids return a graceful
    ``{"error": "not_found", ...}`` payload rather than raising.
    """
    paper = _resolve_paper(arxiv_id_or_id)
    if paper is None:
        return {"error": "not_found", "identifier": str(arxiv_id_or_id)}
    return _paper_detail(paper)


def get_summary(paper_id: str | int) -> dict[str, Any]:
    """Return the stored TL;DR summary and structured LLM insights for one paper."""
    paper = _resolve_paper(paper_id)
    if paper is None:
        return {"error": "not_found", "identifier": str(paper_id)}
    insights = paper.llm_insights if isinstance(paper.llm_insights, dict) else {}
    return {
        "id": paper.id,
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "summary": _truncate(paper.summary_text, _SUMMARY_CHARS),
        "insights": insights,
        "topic_tags": paper.topic_tags_list,
        "llm_relevance_score": paper.llm_relevance_score,
    }


def top_ranked_today(limit: int = _DEFAULT_LIMIT, profile: str | int | None = None) -> dict[str, Any]:
    """Return today's top-ranked fresh papers, mirroring the dashboard feed.

    Reuses the dashboard's ranking order (:func:`rank_score_order_expr`) and its
    daily freshness window (:func:`inbox_freshness_clause`), excluding hidden
    papers. ``profile`` selects the interest-profile context to report against —
    a numeric id, a slug, or a name; ``None`` uses the active profile. The
    resolved profile is echoed so an assistant knows which lens ranked the feed.
    """
    from datetime import timedelta

    from app.services.ranking import rank_score_order_expr
    from app.services.text import now_utc

    n = _clamp_limit(limit)
    resolved_profile = _resolve_profile(profile)

    cutoff = now_utc() - timedelta(days=1)
    papers = (
        Paper.query.filter(Paper.is_hidden.is_(False))
        .filter(inbox_freshness_clause(cutoff))
        .order_by(
            rank_score_order_expr().desc(),
            Paper.publication_dt.desc(),
            Paper.scraped_at.desc(),
        )
        .limit(n)
        .all()
    )
    return {
        "profile": resolved_profile,
        "sort": SortOption.TRENDING.value,
        "count": len(papers),
        "results": [_paper_brief(paper) for paper in papers],
    }


def _resolve_profile(profile: str | int | None) -> dict[str, Any]:
    """Resolve the requested interest profile to a compact dict (default: active).

    Read-only: never changes which profile is active. Falls back to the active
    profile when the selector does not match anything.
    """
    from app.services.profiles import get_active_profile, list_profiles, profile_to_dict

    profiles = list_profiles()
    selected = None
    if profile is not None and not isinstance(profile, bool):
        if isinstance(profile, int) or (isinstance(profile, str) and profile.strip().isdigit()):
            wanted_id = int(profile)
            selected = next((p for p in profiles if p.id == wanted_id), None)
        if selected is None and isinstance(profile, str):
            needle = profile.strip().lower()
            selected = next(
                (p for p in profiles if p.slug.lower() == needle or (p.name or "").lower() == needle),
                None,
            )
    if selected is None:
        selected = get_active_profile()
    return profile_to_dict(selected)


def list_collections() -> dict[str, Any]:
    """List all collections with their paper counts (mirrors ``GET /api/collections``)."""
    paper_count_subquery = (
        db.session.query(
            PaperCollection.collection_id,
            db.func.count(PaperCollection.id).label("paper_count"),
        )
        .group_by(PaperCollection.collection_id)
        .subquery()
    )
    rows = (
        db.session.query(Collection, db.func.coalesce(paper_count_subquery.c.paper_count, 0))
        .outerjoin(paper_count_subquery, Collection.id == paper_count_subquery.c.collection_id)
        .order_by(Collection.name)
        .all()
    )
    collections = [
        {
            "id": collection.id,
            "name": collection.name,
            "description": collection.description or "",
            "color": collection.color,
            "paper_count": int(count),
        }
        for collection, count in rows
    ]
    return {"count": len(collections), "collections": collections}


def ask_paper(paper_id: str | int, question: str) -> dict[str, Any]:
    """Grounded Q&A over ONE paper's own indexed text (see :mod:`app.services.paper_chat`).

    Degrades to ``answer=None`` when no LLM client is configured, returning the
    top-ranked sections so the caller still gets grounded context.
    """
    clean_question = (question or "").strip()
    if not clean_question:
        return {"error": "empty_question"}
    paper = _resolve_paper(paper_id)
    if paper is None:
        return {"error": "not_found", "identifier": str(paper_id)}

    from app.services.paper_chat import answer_paper_question

    try:
        return answer_paper_question(paper.id, clean_question)
    except ValueError:
        return {"error": "not_found", "identifier": str(paper_id)}


# ─────────────────────────── the single mutation ────────────────────────────


def add_to_collection(collection_name_or_id: str | int, paper_id: str | int) -> dict[str, Any]:
    """Add a paper to a collection — the ONE write tool. Idempotent.

    ``collection_name_or_id`` is a numeric id (must already exist) or a name
    (created on first use). Returns ``added=False`` when the paper is already a
    member, so repeated calls converge on the same state. Unknown papers or a
    numeric id with no matching collection return a graceful error.
    """
    paper = _resolve_paper(paper_id)
    if paper is None:
        return {"error": "not_found", "resource": "paper", "identifier": str(paper_id)}

    collection, created = _resolve_or_create_collection(collection_name_or_id)
    if collection is None:
        return {"error": "not_found", "resource": "collection", "identifier": str(collection_name_or_id)}

    existing = PaperCollection.query.filter_by(paper_id=paper.id, collection_id=collection.id).first()
    if existing is not None:
        return {
            "collection_id": collection.id,
            "collection_name": collection.name,
            "paper_id": paper.id,
            "added": False,
            "created_collection": created,
        }

    db.session.add(PaperCollection(paper_id=paper.id, collection_id=collection.id))
    db.session.commit()
    return {
        "collection_id": collection.id,
        "collection_name": collection.name,
        "paper_id": paper.id,
        "added": True,
        "created_collection": created,
    }


def _resolve_or_create_collection(collection_name_or_id: str | int) -> tuple[Collection | None, bool]:
    """Resolve a collection by id (never created) or by name (created on miss).

    Returns ``(collection, created)``; ``(None, False)`` when a numeric id has no
    matching collection.
    """
    if isinstance(collection_name_or_id, bool):
        return None, False
    if isinstance(collection_name_or_id, int):
        return db.session.get(Collection, collection_name_or_id), False

    raw = str(collection_name_or_id or "").strip()
    if not raw:
        return None, False
    if raw.isdigit():
        return db.session.get(Collection, int(raw)), False

    existing = Collection.query.filter_by(name=raw).first()
    if existing is not None:
        return existing, False
    collection = Collection(name=raw)
    db.session.add(collection)
    db.session.commit()
    return collection, True
