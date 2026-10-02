"""MCP tool logic — the testable backend behind :mod:`app.mcp_server`.

Plain functions that expose the personalized, enriched, ranked, full-text-indexed
corpus as compact JSON-serialisable dicts. They use *only* Flask/SQLAlchemy plus
the existing services — **no ``mcp`` SDK import lives here** — so the whole layer
is unit-testable against a seeded DB without the optional MCP extra installed.

Every function runs inside the *current* Flask app context (the MCP wrapper pushes
one per call; tests run inside ``FlaskDBTestCase``'s pushed context). Read paths
are the default; the three write tools (:func:`set_decision`, :func:`tag_papers`,
:func:`add_to_collection`) are clearly separated, never invoked by the read tools,
and each write is validated and logged before it is committed.

Design rules mirrored from the rest of the app:

* Degrade gracefully — an unknown id or a search backend that is unavailable
  returns a structured ``{"error": ...}``/empty payload, never an exception that
  would crash an assistant's tool call.
* Return structured data only (no HTML); the caller renders it.
* Keep dicts compact — assistants pay for every token of context.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

from app.enums import SortOption
from app.models import (
    SCREENING_DECISIONS,
    Collection,
    Paper,
    PaperCollection,
    PaperSection,
    ScrapeRun,
    db,
    in_review_clause,
    inbox_freshness_clause,
)
from app.services.bibtex import _make_cite_key
from app.services.implementation_readiness import implementation_readiness
from app.services.preferences import first_author_name
from app.services.screening import apply_decision
from app.services.text import now_utc

# Bounds so an assistant can never ask for an unbounded result set (each row costs
# context tokens) or a pathological negative/zero limit.
_MAX_LIMIT = 50
_DEFAULT_LIMIT = 10
_ABSTRACT_CHARS = 600
_SUMMARY_CHARS = 1500
# Full-text paging: sections run to ~278k chars, so text is only ever served in pages.
_TEXT_PAGE_CHARS = 8000
_TEXT_MAX_CHARS = 20_000
# Day windows. whats_new is about arrivals, so a quarter is plenty; a collection's
# added-since filter only needs a bound that keeps the date arithmetic in range.
_WHATS_NEW_MAX_DAYS = 90
_ADDED_SINCE_MAX_DAYS = 3650
# What an agent may write. A reason is shown to the owner and a new collection's name to
# everyone, so each is one printable line; a tag is read back by agents, so it is a pattern.
_REASON_CHARS = 200
_NAME_CHARS = 128  # Collection.name
_TAG_RE = re.compile(r"[a-z0-9][a-z0-9 ._-]{0,31}")
# decision_note of a row add_to_collection files: the agent mark, with no decision made.
_AGENT_ADDED_NOTE = "added by agent"
WRITE_LOG_NAME = "mcp_writes.jsonl"

SEARCH_MODES = ("hybrid", "semantic", "keyword")
# get_collection(decision=...): a screening decision, or the rows without one.
DECISION_FILTERS = (*SCREENING_DECISIONS, "unscreened")


def _clamp_limit(limit: int | None, *, default: int = _DEFAULT_LIMIT, most: int = _MAX_LIMIT) -> int:
    try:
        value = int(limit) if limit is not None else default
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return min(value, most)


def _is_row_id(raw: str) -> bool:
    """Whether ``int(raw)`` is a number SQLite can look up as a row id.

    isdigit() will not do: it also passes characters int() rejects (a superscript two),
    and more than 18 digits overflow SQLite's 64-bit INTEGER. Same test as the
    dashboard's ``?ids=``.
    """
    return raw.isdecimal() and len(raw) <= 18


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


def _with_full_text(paper_ids) -> set[int]:
    """The ids among ``paper_ids`` whose paper has extracted sections: one query for all."""
    ids = list(paper_ids)
    if not ids:
        return set()
    rows = db.session.query(PaperSection.paper_id).filter(PaperSection.paper_id.in_(ids)).distinct()
    return {paper_id for (paper_id,) in rows}


def _briefs(papers) -> list[dict[str, Any]]:
    """Compact list-item dicts (search hits, ranked feeds, collection members), in order."""
    papers = list(papers)
    full_text = _with_full_text(paper.id for paper in papers)
    return [
        {
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
            "user_tags": paper.user_tags_list,
            # Spares a get_paper_text call per paper just to learn there is no text.
            "has_full_text": paper.id in full_text,
        }
        for paper in papers
    ]


def _paper_detail(paper: Paper) -> dict[str, Any]:
    """The full single-paper dict (metadata, enrichment summary, tags and collections)."""
    memberships = (
        db.session.query(Collection.id, Collection.name, PaperCollection.decision)
        .join(PaperCollection, PaperCollection.collection_id == Collection.id)
        .filter(PaperCollection.paper_id == paper.id)
        .order_by(Collection.name)
    )
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
        "user_tags": paper.user_tags_list,
        "has_full_text": paper.id in _with_full_text([paper.id]),
        # Why the feed kept it: whitelist match types (or "Interest" / "import") and the terms.
        "match_type": paper.match_type,
        "matched_terms": paper.matched_terms_list,
        # Every collection it has a row in, the ones it was excluded from too (see decision).
        "collections": [{"id": cid, "name": name, "decision": decision} for cid, name, decision in memberships],
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
    if _is_row_id(raw):
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

            # ponytail: over-fetch 3x so skipped (hidden) top hits don't empty the page; if all
            # 3n are hidden the page comes back short. Upgrade: filter hidden ids inside search.
            if normalized_mode == "semantic":
                ordered_ids = [pid for pid, _score in search_semantic(clean_query, top_k=n * 3)]
            else:
                ordered_ids = [row["paper_id"] for row in search_hybrid(clean_query, top_k=n * 3)]
        except Exception:  # noqa: BLE001 — search backends are best-effort; degrade to keyword
            ordered_ids = []

    papers_by_id: dict[int, Paper] = {}
    if ordered_ids:
        # Skipped (hidden) papers drop out of the ranked modes too, as _keyword_ids does.
        papers_by_id = {p.id: p for p in Paper.query.filter(Paper.id.in_(ordered_ids), Paper.is_hidden.is_(False))}
        ordered_ids = [pid for pid in ordered_ids if pid in papers_by_id][:n]
    if not ordered_ids and normalized_mode != "semantic":
        ordered_ids = _keyword_ids(clean_query, n)
        papers_by_id = {p.id: p for p in Paper.query.filter(Paper.id.in_(ordered_ids))}
    results = _briefs(papers_by_id[pid] for pid in ordered_ids)
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
    from app.services.ranking import rank_score_order_expr

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
        "results": _briefs(papers),
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
        if isinstance(profile, int) or (isinstance(profile, str) and _is_row_id(profile.strip())):
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
    """List all collections with their paper counts, excluded papers not counted (mirrors ``GET /api/collections``)."""
    paper_count_subquery = (
        db.session.query(
            PaperCollection.collection_id,
            db.func.count(PaperCollection.id).label("paper_count"),
        )
        .filter(in_review_clause())
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


def get_collection(
    collection_name_or_id: str | int,
    offset: int = 0,
    limit: int = _MAX_LIMIT,
    include_excluded: bool = False,
    tag: str | None = None,
    decision: str | None = None,
    added_since_days: int | None = None,
) -> dict[str, Any]:
    """Return one collection's papers (by id or exact name; never created here).

    Members mirror the collection's Export .bib (hidden and screened-out papers
    excluded, rank order) and carry that export's ``cite_key``, the user's notes
    and the screening ``decision`` (include/maybe/exclude, None = unscreened),
    so an assistant's citations resolve against the same .bib.
    ``include_excluded`` adds the screened-out papers back. Paged by
    ``offset``/``limit`` (``limit`` capped at 50); follow ``next_offset``.

    Three optional filters narrow the list, and ``count`` with it. ``tag`` keeps the
    papers that carry exactly that user tag. ``decision`` keeps one screening state
    (one of DECISION_FILTERS; ``exclude`` implies ``include_excluded``, and any other
    value is an error, never an unfiltered list). ``added_since_days`` keeps the
    papers filed here in the last N days: this is how a reader sees what a query
    refresh added. It is clamped to 1..3650, and the result names the window applied.
    """
    from app.services.ranking import rank_score_order_expr

    if decision is not None and decision not in DECISION_FILTERS:
        return {"error": "invalid_decision", "decision": str(decision), "allowed": list(DECISION_FILTERS)}
    collection, _created = _resolve_or_create_collection(collection_name_or_id, create=False)
    if collection is None:
        return {"error": "not_found", "resource": "collection", "identifier": str(collection_name_or_id)}

    start = max(0, offset)
    query = (
        db.session.query(Paper, PaperCollection.decision)
        .join(PaperCollection, PaperCollection.paper_id == Paper.id)
        .filter(PaperCollection.collection_id == collection.id, Paper.is_hidden.is_(False))
        .order_by(rank_score_order_expr().desc(), Paper.id)
    )
    if not include_excluded and decision != "exclude":
        query = query.filter(in_review_clause())
    if decision == "unscreened":
        query = query.filter(PaperCollection.decision.is_(None))
    elif decision is not None:
        query = query.filter(PaperCollection.decision == decision)
    if tag is not None:
        # One whole element of the stored JSON list: json.dumps supplies the quotes around
        # it. instr rather than LIKE: tags differ by case, and a tag may hold % or _ or be
        # longer than a LIKE pattern may be.
        # ponytail: a substring test on the JSON text, so a search for x also finds a tag
        # that ends in "x, and a search for ", " finds every paper with two tags or more.
        # Upgrade: json_each equality, behind json_valid (legacy rows may not be JSON).
        query = query.filter(db.func.instr(Paper.user_tags, json.dumps(str(tag))) > 0)
    window = {}
    if added_since_days is not None:
        days = _clamp_limit(added_since_days, default=1, most=_ADDED_SINCE_MAX_DAYS)
        query = query.filter(PaperCollection.added_at >= now_utc() - timedelta(days=days))
        # Said back, because 0, a negative or a huge value is clamped rather than refused.
        window = {"added_since_days": days}
    total = query.count()
    rows = query.offset(start).limit(_clamp_limit(limit, default=_MAX_LIMIT)).all()
    end = start + len(rows)
    return {
        "id": collection.id,
        "name": collection.name,
        "description": collection.description or "",
        "count": total,
        "offset": start,
        "next_offset": end if end < total else None,
        **window,
        "papers": [
            {
                **brief,
                "cite_key": _make_cite_key(paper),
                "user_notes": paper.user_notes or "",
                "decision": screened,
            }
            for brief, (paper, screened) in zip(_briefs(paper for paper, _screened in rows), rows)
        ],
    }


def whats_new(since_days: int = 7, collection: str | int | None = None, limit: int = 20) -> dict[str, Any]:
    """Papers that arrived lately and sit close to a collection without being in one.

    Candidates to screen, not recommendations. They are the visible papers scraped in
    the last ``since_days`` (at most 90) that are in review in no collection.
    Each is scored from its stored vector by the collection scorer the feed uses (no
    embedding model is loaded here) and attributed to its best collection among those
    it has no row in, so an exclusion is remembered for that one collection only.
    Results clear the admission floor (z >= AFFINITY_Z_MIN), best z first; ``limit``
    (capped at 50) is a total over all collections, and ``collection`` (id or exact
    name, never created) keeps the papers attributed to that one.

    The header says how fresh the corpus is (``last_scrape``), how many candidates
    there were (``arrived``), how many of them have no stored vector yet
    (``unscored``) and how many pass per collection (``by_collection``), so a reader
    knows where to look closer with ``collection=``. Without a collection profile,
    or with a vector index that cannot be read, the result is empty and ``reason``
    says why.
    """
    import numpy as np
    from flask import current_app

    from app.services.embeddings import get_embedding_service
    from app.services.interest_model import (
        AFFINITY_Z_MIN,
        MIN_BACKGROUND,
        MIN_POSITIVE_FEEDBACK,
        affinity_scores,
        build_interest_profile,
        nearest_member,
    )

    only = None
    if collection is not None:
        only, _created = _resolve_or_create_collection(collection, create=False)
        if only is None:
            return {"error": "not_found", "resource": "collection", "identifier": str(collection)}

    days = _clamp_limit(since_days, default=7, most=_WHATS_NEW_MAX_DAYS)
    last_scrape = None
    last_run = ScrapeRun.query.order_by(ScrapeRun.started_at.desc()).first()
    if last_run is not None:
        finished = last_run.finished_at  # stored as naive UTC; None while the run is going
        last_scrape = {
            "status": last_run.status,
            "finished_at": finished.isoformat(timespec="seconds") + "Z" if finished else None,
        }
    # ponytail: a paper already in review in one collection is not offered to another.
    # Upgrade: drop this filter and score every collection the paper has no row in.
    in_review = db.session.query(PaperCollection.paper_id).filter(in_review_clause())
    candidates = db.session.query(Paper.id).filter(
        Paper.is_hidden.is_(False),
        Paper.scraped_at >= now_utc() - timedelta(days=days),
        Paper.id.not_in(in_review),
    )
    candidate_ids = [paper_id for (paper_id,) in candidates]
    result: dict[str, Any] = {
        "since_days": days,
        # A run that fetched nothing still records "success": `arrived` is the real signal.
        "last_scrape": last_scrape,
        "arrived": len(candidate_ids),
        "unscored": len(candidate_ids),
        "by_collection": [],
        "count": 0,
        "results": [],
    }
    try:
        service = get_embedding_service()
        scored_ids, vectors = service.get_paper_vectors(candidate_ids)
    except Exception:  # noqa: BLE001 — as in search_papers: a backend that is down is an answer, not a crash
        result["reason"] = "The vector index could not be read, so nothing was scored."
        return result
    result["unscored"] -= len(scored_ids)

    profile = build_interest_profile(current_app._get_current_object())
    if profile is None or profile.centroids is None:
        result["reason"] = (
            f"No collection can be scored yet: that takes a collection with at least {MIN_POSITIVE_FEEDBACK} "
            f"embedded papers in review and {MIN_BACKGROUND} embedded papers outside every collection."
        )
        return result
    column = {collection_id: index for index, collection_id in enumerate(profile.collection_ids)}
    if only is not None and only.id not in column:
        result["reason"] = (
            f"Nothing can be scored against {only.name!r}: it has fewer than {MIN_POSITIVE_FEEDBACK} "
            "embedded papers in review."
        )
        return result

    z = affinity_scores(profile, vectors)
    # A candidate is in review nowhere, so any row it has is an exclusion: that
    # collection is not offered again, the others still are.
    row = {paper_id: index for index, paper_id in enumerate(scored_ids)}
    for paper_id, collection_id in db.session.query(PaperCollection.paper_id, PaperCollection.collection_id).filter(
        PaperCollection.decision == "exclude"
    ):
        if paper_id in row and collection_id in column:
            z[row[paper_id], column[collection_id]] = -np.inf
    best = z.argmax(axis=1)
    best_z = z[np.arange(len(best)), best]
    passing = np.flatnonzero(best_z >= AFFINITY_Z_MIN)  # false for NaN too: a broken vector never passes
    counts = np.bincount(best[passing], minlength=len(column))
    result["by_collection"] = [
        {"id": collection_id, "name": name, "passing": int(count)}
        for collection_id, name, count in zip(profile.collection_ids, profile.labels, counts)
    ]
    if only is not None:
        passing = passing[best[passing] == column[only.id]]
    # ponytail: no paging, so of more than 50 passing papers only the best 50 show
    # (per collection with collection=). Upgrade: an offset, as get_collection has.
    top = passing[np.argsort(-best_z[passing], kind="stable")][: _clamp_limit(limit, default=20)].tolist()
    if not top:
        return result

    # The member each result sits closest to, among the visible members get_collection lists.
    nearest: dict[int, int] = {}  # row of `vectors` -> paper id of its nearest member
    for won in sorted({int(best[index]) for index in top}):
        rows = [index for index in top if best[index] == won]
        members = (
            db.session.query(PaperCollection.paper_id)
            .join(Paper, Paper.id == PaperCollection.paper_id)
            .filter(
                PaperCollection.collection_id == profile.collection_ids[won],
                in_review_clause(),
                Paper.is_hidden.is_(False),
            )
        )
        member_ids, member_vectors = service.get_paper_vectors([paper_id for (paper_id,) in members])
        if member_ids:
            for index, member in zip(rows, nearest_member(profile, vectors[rows], member_vectors)):
                nearest[index] = member_ids[member]

    wanted = [scored_ids[index] for index in top] + list(nearest.values())
    papers = {paper.id: paper for paper in Paper.query.filter(Paper.id.in_(wanted))}
    for brief, index in zip(_briefs(papers[scored_ids[index]] for index in top), top):
        member = papers.get(nearest.get(index))
        result["results"].append(
            {
                **brief,
                "z": round(float(best_z[index]), 2),
                "collection": {"id": profile.collection_ids[best[index]], "name": profile.labels[best[index]]},
                "nearest_member": member and {"arxiv_id": member.arxiv_id, "title": member.title},
            }
        )
    result["count"] = len(top)
    return result


def get_paper_text(
    paper_id: str | int, order_index: int | None = None, offset: int = 0, max_chars: int = _TEXT_PAGE_CHARS
) -> dict[str, Any]:
    """Read one paper's extracted full text verbatim, a page at a time.

    Without ``order_index``: the section table of contents (``order_index``,
    ``section_type``, ``chars``) plus the full abstract. With it: that section's
    ``text[offset:offset + max_chars]`` (``max_chars`` capped at 20k) and a
    ``next_offset`` to continue from (``None`` at the end). Papers never
    extracted return ``has_full_text=False`` and an empty contents list.
    """
    paper = _resolve_paper(paper_id)
    if paper is None:
        return {"error": "not_found", "identifier": str(paper_id)}
    head = {"id": paper.id, "arxiv_id": paper.arxiv_id, "title": paper.title, "link": paper.link}

    if order_index is None:
        rows = (
            db.session.query(PaperSection.order_index, PaperSection.section_type, db.func.length(PaperSection.text))
            .filter(PaperSection.paper_id == paper.id)
            .order_by(PaperSection.order_index)
            .all()
        )
        sections = [{"order_index": idx, "section_type": kind, "chars": int(chars or 0)} for idx, kind, chars in rows]
        return {
            **head,
            "has_full_text": bool(sections),
            "abstract": (paper.abstract_text or "").strip(),
            "sections": sections,
        }

    section = PaperSection.query.filter_by(paper_id=paper.id, order_index=order_index).first()
    if section is None:
        return {"error": "not_found", "resource": "section", "identifier": f"{paper.id}:{order_index}"}
    text = section.text or ""
    start = max(0, offset)
    page = min(max_chars, _TEXT_MAX_CHARS) if max_chars > 0 else _TEXT_PAGE_CHARS
    end = min(start + page, len(text))
    return {
        **head,
        "order_index": section.order_index,
        "section_type": section.section_type,
        "total_chars": len(text),
        "offset": start,
        "text": text[start:end],
        "next_offset": end if end < len(text) else None,
    }


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


# ─────────────────────────── write tools ────────────────────────────
# For attended sessions: `cv-arxiv-mcp --read-only` registers none of them. Each is
# narrow, validates what the agent sends, and logs the write before committing it.


def _one_line(text: object, most: int) -> str | None:
    """``text`` stripped, if it is one printable line of 1..``most`` characters; else None.

    ``str.isprintable`` turns down newlines and control characters, and with them the
    invisible format characters (zero-width, bidi overrides).
    """
    cleaned = text.strip() if isinstance(text, str) else ""
    return cleaned if 0 < len(cleaned) <= most and cleaned.isprintable() else None


def _commit_logged(tool: str, **entry: Any) -> dict[str, Any] | None:
    """Flush the pending write, append its line to the write log, commit; None when it went through.

    The line is written before the commit, so nothing is committed that the log does
    not show (a line whose commit then failed describes a write that did not happen).
    Any failure rolls back and comes back as an error payload: a locked database, or a
    log that cannot be written, must not raise out of a tool.
    """
    try:
        db.session.flush()
        # Next to the database file, not in app.instance_path: the log belongs to that
        # database, and every test's temporary database gets its own.
        log_path = Path(db.engine.url.database).with_name(WRITE_LOG_NAME)
        line = json.dumps({"time": now_utc().isoformat(timespec="seconds") + "Z", "tool": tool, **entry})
        with log_path.open("a", encoding="utf-8") as log:
            log.write(line + "\n")
        db.session.commit()
    except Exception as exc:  # noqa: BLE001 — a database error and an OS error alike
        db.session.rollback()
        return {"error": "write_failed", "cause": type(exc).__name__}
    return None


def set_decision(collection: str | int, paper: str | int, decision: str, reason: str) -> dict[str, Any]:
    """Record an agent's screening decision for ONE paper in a collection, with its reason.

    ``decision`` is one of SCREENING_DECISIONS and ``reason`` one printable line of at
    most 200 characters. The collection is resolved by id or exact name and never
    created. A paper with no row in it yet is filed with the decision: an ``exclude``
    is how a ``whats_new`` candidate is turned down, so it is not offered again.

    The reason is stored as the row's ``decision_note``, which marks the decision as
    the agent's until the owner confirms or overrules it in the web UI. No read tool
    returns it, and neither does this one. A decision the owner made (one without that
    mark) is refused, never overwritten; the agent's own earlier one may be changed.
    Returns the previous decision (None: unscreened, or no row at all when
    ``created``). There is no undo here: that is the owner's, in the web UI.
    """
    if decision not in SCREENING_DECISIONS:
        return {"error": "invalid_decision", "decision": str(decision), "allowed": list(SCREENING_DECISIONS)}
    note = _one_line(reason, _REASON_CHARS)
    if note is None:
        return {"error": "invalid_reason", "expected": f"one printable line of 1 to {_REASON_CHARS} characters"}
    found = _resolve_paper(paper)
    if found is None:
        return {"error": "not_found", "resource": "paper", "identifier": str(paper)}
    target, _created = _resolve_or_create_collection(collection, create=False)
    if target is None:
        return {"error": "not_found", "resource": "collection", "identifier": str(collection)}

    # ponytail: check, then write, in two statements. An owner click that lands between
    # them (in the web process) is overwritten by this decision, which stays marked as
    # the agent's. Upgrade: one UPDATE guarded by the same condition.
    row = PaperCollection.query.filter_by(paper_id=found.id, collection_id=target.id).first()
    if row is not None and row.decision is not None and row.decision_note is None:
        return {"error": "owner_decision", "collection_id": target.id, "paper_id": found.id, "decision": row.decision}
    previous = apply_decision(target.id, [found.id], decision, note=note, create=True)
    result = {
        "collection_id": target.id,
        "collection_name": target.name,
        "paper_id": found.id,
        "arxiv_id": found.arxiv_id,
        "title": found.title,  # said back: a mistyped id is a real paper more often than not
        "decision": decision,
        "previous": previous.get(found.id),
        "created": found.id not in previous,
    }
    failed = _commit_logged(
        "set_decision",
        collection_id=result["collection_id"],
        paper_ids=[result["paper_id"]],
        previous=result["previous"],
        value=decision,
        created=result["created"],
        reason=note,
    )
    return failed or result


def tag_papers(papers: list[str | int], tag: str) -> dict[str, Any]:
    """Add one user tag to up to 50 papers. Add-only: no tool removes a tag.

    ``tag`` must match ``[a-z0-9][a-z0-9 ._-]{0,31}`` in full: agents read tags back
    (every listed paper carries its ``user_tags``), so free text is not accepted. All
    or nothing: when any id is unknown nothing is tagged, and the unknown ids come
    back. Papers that carry the tag already are left alone (``already_tagged``).
    """
    cleaned = tag.strip() if isinstance(tag, str) else ""
    if not _TAG_RE.fullmatch(cleaned):
        return {"error": "invalid_tag", "pattern": _TAG_RE.pattern}
    if not isinstance(papers, list) or not 0 < len(papers) <= _MAX_LIMIT:
        return {"error": "invalid_papers", "expected": f"a list of 1 to {_MAX_LIMIT} paper ids"}
    resolved = [(identifier, _resolve_paper(identifier)) for identifier in papers]
    unknown = [str(identifier) for identifier, found in resolved if found is None]
    if unknown:
        return {"error": "not_found", "resource": "paper", "identifiers": unknown}

    # One paper may be named twice (by row id and by arXiv id): tag it once.
    unique = {found.id: found for _identifier, found in resolved}
    # ponytail: a read-modify-write of each paper's JSON list, outside the lock the REST
    # tag routes take (_TAG_WRITE_LOCK in routes/api/papers.py is process-local, and
    # this is another process): of two tag edits that reach the same paper at the same
    # moment, one can be lost. Upgrade: a single-statement JSON update (json_insert /
    # json_remove) in both writers.
    tagged = [found for found in unique.values() if cleaned not in found.user_tags_list]
    for found in tagged:
        found.user_tags = [*found.user_tags_list, cleaned]
    result = {
        "tag": cleaned,
        "tagged": [{"id": found.id, "arxiv_id": found.arxiv_id, "title": found.title} for found in tagged],
        "already_tagged": len(unique) - len(tagged),
    }
    if not tagged:
        return result
    return _commit_logged("tag_papers", paper_ids=[row["id"] for row in result["tagged"]], value=cleaned) or result


def add_to_collection(collection_name_or_id: str | int, paper_id: str | int, create: bool = False) -> dict[str, Any]:
    """File a paper into a collection. Idempotent.

    ``collection_name_or_id`` is a numeric id or an exact name. An unknown name is an
    error, not a new collection (a mistyped name used to create one silently), unless
    ``create`` is set and the name is one printable line; a number is always an id and
    is never created. Returns ``added=False`` when the paper already has a row there
    (an excluded one too), so repeated calls converge on the same state. The row it
    adds is unscreened and carries the agent mark (``decision_note``) until the owner
    screens it.
    """
    paper = _resolve_paper(paper_id)
    if paper is None:
        return {"error": "not_found", "resource": "paper", "identifier": str(paper_id)}

    collection, created = _resolve_or_create_collection(collection_name_or_id, create=False)
    if collection is None and create and _one_line(collection_name_or_id, _NAME_CHARS):
        # ponytail: the helper commits the new collection on its own, before the logged
        # write below; if that write then fails, an empty collection is left without a
        # log line. Upgrade: create it in the same transaction.
        collection, created = _resolve_or_create_collection(collection_name_or_id)
    if collection is None:
        return {"error": "not_found", "resource": "collection", "identifier": str(collection_name_or_id)}

    added = PaperCollection.query.filter_by(paper_id=paper.id, collection_id=collection.id).first() is None
    result = {
        "collection_id": collection.id,
        "collection_name": collection.name,
        "paper_id": paper.id,
        "added": added,
        "created_collection": created,
    }
    if not added:
        return result
    apply_decision(collection.id, [paper.id], None, note=_AGENT_ADDED_NOTE, create=True)
    failed = _commit_logged(
        "add_to_collection",
        collection_id=result["collection_id"],
        paper_ids=[result["paper_id"]],
        created=True,
        created_collection=created,
    )
    return failed or result


def _resolve_or_create_collection(
    collection_name_or_id: str | int, *, create: bool = True
) -> tuple[Collection | None, bool]:
    """Resolve a collection by id (never created) or by name (created on miss).

    Returns ``(collection, created)``; ``(None, False)`` when a numeric id has no
    matching collection, or a name misses with ``create=False`` (read-only lookup).
    """
    if isinstance(collection_name_or_id, bool):
        return None, False
    if isinstance(collection_name_or_id, int):
        return db.session.get(Collection, collection_name_or_id), False

    raw = str(collection_name_or_id or "").strip()
    if not raw:
        return None, False
    if raw.isdigit():
        # A number is an id and never a name, also one that cannot be a row id: no
        # collection is created for it.
        return (db.session.get(Collection, int(raw)) if _is_row_id(raw) else None), False

    existing = Collection.query.filter_by(name=raw).first()
    if existing is not None or not create:
        return existing, False
    collection = Collection(name=raw)
    db.session.add(collection)
    db.session.commit()
    return collection, True
