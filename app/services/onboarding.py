"""Ranking-onboarding helpers: cold-start bootstrap + active-learning selection.

Two flows accelerate the learned interest profile (see
``app.services.interest_model``) for a fresh install:

* :func:`bootstrap_from_arxiv_ids` ingests a pasted list of arXiv IDs as
  implicit "saves" so the profile activates immediately (it needs
  ``MIN_POSITIVE_FEEDBACK`` saved-with-embeddings papers).
* :func:`select_uncertain_papers` surfaces 1-2 boundary papers — the ones whose
  similarity to the positive centroid sits nearest the middle of the candidate
  range — to gather the most informative feedback.

Single-paper arXiv fetch lives here (there is no shared backend for it yet); it
reuses the project's HTTP client and the ingest parsing helpers.
"""

from __future__ import annotations

import logging
import re

import feedparser

from app.services.embeddings import get_embedding_service
from app.services.http_client import request_with_backoff
from app.services.ingest.base import (
    clean_abstract,
    extract_author_names,
    parse_publication_dt,
)
from app.services.text import clean_whitespace

LOGGER = logging.getLogger(__name__)

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ARXIV_API_TIMEOUT = 45
_ARXIV_API_ATTEMPTS = 4
_ARXIV_API_BASE_DELAY = 2.0

# Matches a bare arXiv id (new "2401.01234" or old "math.GT/0309136" scheme),
# stripping any leading "arXiv:" label, surrounding abs/pdf URL, or trailing
# version suffix ("v2") so the same paper normalizes to one canonical id. The
# legacy archive name may contain hyphens (``cond-mat``, ``q-bio``) and an
# optional subject-class component (``.GT``, ``.str-el``) that is captured
# separately so it can be dropped — see normalize_arxiv_id.
_NEW_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")
_OLD_ID_RE = re.compile(r"([a-z][a-z\-]*)(?:\.[a-z][a-z\-]*)?/(\d{7})", re.IGNORECASE)


def normalize_arxiv_id(raw: str) -> str | None:
    """Normalize a raw arXiv reference to a canonical id (no version/URL/prefix).

    Accepts bare ids, ``arXiv:`` prefixes, and abs/pdf URLs in either the new
    (``2401.01234``) or legacy (``cond-mat.str-el/0309136``) scheme. Returns
    ``None`` when no id can be extracted.

    For legacy ids the canonical identifier is ``archive/NNNNNNN`` — the subject
    class (``.GT``, ``.str-el``) is metadata, not part of the id the arXiv API
    ``id_list`` query resolves — so it is stripped and the archive lowercased
    (``cond-mat.str-el/0309136`` -> ``cond-mat/0309136``).
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    # Drop a trailing version suffix early so it never leaks into the result.
    text = re.sub(r"v\d+$", "", text)

    new_match = _NEW_ID_RE.search(text)
    if new_match:
        return new_match.group(1)
    old_match = _OLD_ID_RE.search(text)
    if old_match:
        return f"{old_match.group(1).lower()}/{old_match.group(2)}"
    return None


def _build_embed_text(title: str, abstract: str) -> str:
    """Match ``scrape_engine._generate_embeddings``' title + abstract concat."""
    return f"{title} {abstract or ''}"


def fetch_arxiv_metadata(arxiv_ids: list[str]) -> list[dict]:
    """Fetch metadata for ``arxiv_ids`` via the arXiv API ``id_list`` query.

    Returns a list of entry dicts (one per resolved id) with keys
    ``arxiv_id, title, authors, abstract, link, pdf_link, categories,
    published, publication_dt, publication_date``. Ids the API does not resolve
    are simply absent from the result.
    """
    normalized = [nid for nid in (normalize_arxiv_id(raw) for raw in arxiv_ids) if nid]
    if not normalized:
        return []

    response = request_with_backoff(
        "GET",
        _ARXIV_API_URL,
        params={"id_list": ",".join(normalized), "max_results": len(normalized)},
        timeout=_ARXIV_API_TIMEOUT,
        attempts=_ARXIV_API_ATTEMPTS,
        base_delay=_ARXIV_API_BASE_DELAY,
        rate_limit_profile="bulk",
    )
    feed = feedparser.parse(response.content)

    entries: list[dict] = []
    for entry in feed.entries:
        link = getattr(entry, "id", "") or getattr(entry, "link", "")
        arxiv_id = normalize_arxiv_id(link) or normalize_arxiv_id(getattr(entry, "title", ""))
        if not arxiv_id:
            continue
        published = getattr(entry, "published", None)
        publication_dt, publication_date = parse_publication_dt(published)
        categories = [
            term for term in (tag.get("term", "").strip() for tag in getattr(entry, "tags", []) or []) if term
        ]
        pdf_link = ""
        for resource in getattr(entry, "links", []) or []:
            if resource.get("type") == "application/pdf" or resource.get("title") == "pdf":
                pdf_link = resource.get("href", "")
                break
        abs_link = f"https://arxiv.org/abs/{arxiv_id}"
        entries.append(
            {
                "arxiv_id": arxiv_id,
                "title": clean_whitespace(getattr(entry, "title", "")),
                "authors": extract_author_names(entry),
                "abstract": clean_abstract(getattr(entry, "summary", "")),
                "link": abs_link,
                "pdf_link": pdf_link or f"https://arxiv.org/pdf/{arxiv_id}",
                "categories": categories,
                "published": published,
                "publication_dt": publication_dt,
                "publication_date": publication_date,
            }
        )
    return entries


def _resolve_app(app):
    if app is not None:
        return app
    from flask import current_app

    return current_app._get_current_object()


def _find_existing_paper(arxiv_id: str):
    from app.models import Paper, db

    return db.session.query(Paper).filter_by(arxiv_id=arxiv_id).first()


def _insert_paper(entry: dict):
    """Insert a Paper from a fetched entry, mirroring ``_save_results`` mapping."""
    from app.models import Paper, db
    from app.services.text import now_utc

    now = now_utc()
    authors = entry.get("authors") or []
    paper = Paper(
        arxiv_id=entry["arxiv_id"],
        title=entry.get("title") or entry["arxiv_id"],
        authors=", ".join(authors),
        link=entry["link"],
        pdf_link=entry["pdf_link"],
        abstract_text=entry.get("abstract", ""),
        categories=entry.get("categories", []),
        match_type="Bootstrap",
        matched_terms=[],
        paper_score=0.0,
        publication_date=entry.get("publication_date") or "Date Unknown",
        publication_dt=entry.get("publication_dt"),
        scraped_date=now.date().isoformat(),
        scraped_at=now,
    )
    db.session.add(paper)
    db.session.commit()
    return paper


def _embed_paper(app, paper) -> None:
    """Add ``paper`` to the FAISS index (best-effort; non-fatal on failure)."""
    try:
        service = get_embedding_service(app)
        text = _build_embed_text(paper.title, paper.abstract_text)
        service.add_papers([paper.id], [text])
        # Persist immediately: without save() the vector lives only in the
        # in-process singleton and is silently dropped by the next
        # reset_embedding_service() (concurrent scrape / backup restore) or a
        # restart, leaving the cold-start interest profile un-seeded.
        service.save()
    except Exception:
        LOGGER.warning("Bootstrap embedding failed for paper %s (non-fatal)", paper.id, exc_info=True)


def _saved_paper_count() -> int:
    from app.models import PaperFeedback, db

    return int(
        db.session.query(db.func.count(db.func.distinct(PaperFeedback.paper_id)))
        .filter(PaperFeedback.action == "save")
        .scalar()
        or 0
    )


def _has_save_feedback(paper_id: int) -> bool:
    """True when ``paper_id`` already carries a ``save`` feedback row."""
    from app.models import PaperFeedback, db

    return (
        db.session.query(PaperFeedback.id)
        .filter(PaperFeedback.paper_id == paper_id, PaperFeedback.action == "save")
        .first()
        is not None
    )


def bootstrap_from_arxiv_ids(arxiv_ids: list[str], *, app=None) -> dict:
    """Ingest ``arxiv_ids`` as implicit saves to seed the interest profile.

    For each id: normalize, dedupe/insert the Paper, embed it, and mark a
    ``save`` via the feedback service. After the loop the interest similarities
    are recomputed (falls back to a plain score recompute while the profile is
    still inactive).

    Returns a summary dict::

        {"requested", "ingested", "already_present", "failed",
         "profile_active", "saved_total"}
    """
    from app.services.feedback import apply_feedback_action
    from app.services.interest_model import build_interest_profile, recompute_interest_similarities

    app = _resolve_app(app)

    requested = [nid for nid in (normalize_arxiv_id(raw) for raw in (arxiv_ids or [])) if nid]
    # Preserve order while dropping duplicates from the pasted list itself.
    unique_ids = list(dict.fromkeys(requested))

    summary = {
        "requested": len(arxiv_ids or []),
        "ingested": [],
        "already_present": [],
        "failed": [],
        "profile_active": False,
        "saved_total": 0,
    }
    if not unique_ids:
        return summary

    # Resolve metadata only for ids that aren't already stored.
    to_fetch = [aid for aid in unique_ids if _find_existing_paper(aid) is None]
    fetched_by_id: dict[str, dict] = {}
    if to_fetch:
        try:
            fetched_by_id = {entry["arxiv_id"]: entry for entry in fetch_arxiv_metadata(to_fetch)}
        except Exception:
            LOGGER.warning("Bootstrap arXiv fetch failed (non-fatal)", exc_info=True)
            fetched_by_id = {}

    for arxiv_id in unique_ids:
        paper = _find_existing_paper(arxiv_id)
        already_present = paper is not None
        if paper is None:
            entry = fetched_by_id.get(arxiv_id)
            if entry is None:
                summary["failed"].append(arxiv_id)
                continue
            try:
                paper = _insert_paper(entry)
            except Exception:
                LOGGER.warning("Bootstrap insert failed for %s (non-fatal)", arxiv_id, exc_info=True)
                summary["failed"].append(arxiv_id)
                continue

        _embed_paper(app, paper)
        # apply_feedback_action TOGGLES, so calling it on a paper that is already
        # saved would UNSAVE it (and corrupt saved_total). Only add a save when the
        # paper does not already carry one — re-running bootstrap or pasting an
        # already-saved id is then idempotent.
        if not _has_save_feedback(paper.id):
            try:
                apply_feedback_action(paper.id, "save")
            except Exception:
                LOGGER.warning("Bootstrap save failed for %s (non-fatal)", arxiv_id, exc_info=True)
                summary["failed"].append(arxiv_id)
                continue

        if already_present:
            summary["already_present"].append(arxiv_id)
        else:
            summary["ingested"].append(arxiv_id)

    # Rebuild the profile + rescore (no-op-ish until enough saves accumulate).
    try:
        recompute_interest_similarities(app)
    except Exception:
        LOGGER.warning("Bootstrap recompute failed (non-fatal)", exc_info=True)

    summary["profile_active"] = build_interest_profile(app) is not None
    summary["saved_total"] = _saved_paper_count()
    return summary


def _papers_with_feedback() -> set[int]:
    """Paper ids carrying any save/priority/skip/ignore feedback row."""
    from app.services.interest_model import (
        NEGATIVE_ACTIONS,
        POSITIVE_ACTIONS,
        _paper_ids_for_actions,
    )

    return set(_paper_ids_for_actions(POSITIVE_ACTIONS + NEGATIVE_ACTIONS))


def select_uncertain_papers(*, limit: int = 2, min_saves: int = 3) -> list[dict]:
    """Return up to ``limit`` "uncertain" boundary papers for active learning.

    Builds a positive centroid (L2-normalized mean of saved papers' vectors —
    independent of the interest-profile gate, so it works below
    ``MIN_POSITIVE_FEEDBACK``). Candidates are indexed papers with no feedback;
    the most ambiguous are those whose centroid similarity sits nearest the
    midpoint of the candidate similarity range. Returns ``[]`` when fewer than
    ``min_saves`` papers are saved or no scorable candidate exists.
    """
    import numpy as np

    from app.models import Paper, PaperFeedback, db

    saved_ids = [
        row[0]
        for row in db.session.query(PaperFeedback.paper_id).filter(PaperFeedback.action == "save").distinct().all()
    ]
    if len(saved_ids) < min_saves:
        return []

    service = get_embedding_service()
    # Only the vectors are needed (for the centroid); the found-ids list is unused.
    _found_saved, saved_vectors = service.get_paper_vectors(saved_ids)
    if saved_vectors.shape[0] < min_saves:
        return []

    centroid = saved_vectors.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        return []
    centroid = (centroid / norm).astype(np.float32)

    feedback_ids = _papers_with_feedback()
    candidate_ids = [
        row[0] for row in db.session.query(Paper.id).order_by(Paper.id).all() if row[0] not in feedback_ids
    ]
    if not candidate_ids:
        return []

    found_candidates, candidate_vectors = service.get_paper_vectors(candidate_ids)
    if not found_candidates:
        return []

    sims = candidate_vectors @ centroid
    midpoint = (float(sims.min()) + float(sims.max())) / 2.0
    # Most ambiguous = closest to the midpoint of the similarity range. Tie-break
    # on paper id (found_candidates is id-ordered) so selection is deterministic.
    order = sorted(range(len(found_candidates)), key=lambda i: (abs(float(sims[i]) - midpoint), found_candidates[i]))

    selected_ids = [found_candidates[i] for i in order[: max(0, limit)]]
    sim_by_id = {found_candidates[i]: float(sims[i]) for i in range(len(found_candidates))}

    papers = {p.id: p for p in db.session.query(Paper).filter(Paper.id.in_(selected_ids)).all()}
    results: list[dict] = []
    for pid in selected_ids:
        paper = papers.get(pid)
        if paper is None:
            continue
        results.append(
            {
                "paper_id": pid,
                "title": paper.title,
                "authors": paper.authors,
                "similarity": round(sim_by_id[pid], 4),
            }
        )
    return results
