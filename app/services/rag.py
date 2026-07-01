"""Conversational RAG over the user's saved corpus.

Retrieval is grounded in the papers the reader explicitly saved (``PaperFeedback``
rows with ``action == "save"``). Hybrid search runs over the full corpus, so we
filter its results down to the saved set before building the context block.

Synthesis is optional: when the LLM is disabled or no API key is available we
return the retrieved sources plus an extractive context block with
``synthesis=None`` and ``llm_used=False``. The network is only touched when a
client is successfully built, and any failure degrades back to ``synthesis=None``.
"""

from __future__ import annotations

import logging

from flask import current_app

from app.enums import FeedbackAction
from app.models import Paper, PaperFeedback, PaperSection, db
from app.services.search import search_hybrid

LOGGER = logging.getLogger(__name__)

# Per-paper character budget for the extractive context block. Keeps the prompt
# (and the extractive payload returned when synthesis is off) within a sane size.
_CONTEXT_CHAR_BUDGET = 1200
_SNIPPET_CHAR_BUDGET = 320

_SYSTEM_PROMPT = (
    "You are a research assistant answering questions about a reader's saved "
    "papers. Answer ONLY using the provided context. Cite the relevant paper "
    "titles inline. If the context does not contain enough information to answer, "
    "say so plainly rather than guessing."
)


def _saved_paper_ids() -> set[int]:
    """Return the ids of papers the reader has saved."""
    rows = db.session.query(PaperFeedback.paper_id).filter(PaperFeedback.action == FeedbackAction.SAVE.value).all()
    return {row[0] for row in rows}


def _best_section_text(paper: Paper, query: str) -> str:
    """Return the most query-relevant section text for ``paper`` (empty if none)."""
    try:
        from app.services.embeddings import get_embedding_service

        service = get_embedding_service()
        hits = service.search_sections(query, top_k=12)
    except Exception:  # noqa: BLE001 — section ranking is best-effort; degrade to none
        hits = []

    best_type: str | None = None
    for hit in hits:
        if hit.get("paper_id") == paper.id:
            best_type = hit.get("section_type")
            break

    sections = paper.sections
    if best_type is not None:
        match = sections.filter_by(section_type=best_type).first()
        if match is not None and match.text:
            return match.text

    # Fall back to the first ordered section so the context is still useful.
    first = sections.order_by(PaperSection.order_index).first()
    if first is not None and first.text:
        return first.text
    return ""


def _build_paper_block(paper: Paper, query: str) -> str:
    """Assemble a truncated context block (title + abstract + best section)."""
    parts: list[str] = [f"Title: {paper.title}"]
    abstract = (paper.abstract_text or paper.summary_text or "").strip()
    if abstract:
        parts.append(f"Abstract: {abstract}")
    section_text = _best_section_text(paper, query).strip()
    if section_text:
        parts.append(f"Excerpt: {section_text}")
    block = "\n".join(parts)
    if len(block) > _CONTEXT_CHAR_BUDGET:
        block = block[: _CONTEXT_CHAR_BUDGET - 1].rstrip() + "…"
    return block


def _snippet(paper: Paper) -> str:
    text = (paper.abstract_text or paper.summary_text or "").strip()
    if len(text) > _SNIPPET_CHAR_BUDGET:
        return text[: _SNIPPET_CHAR_BUDGET - 1].rstrip() + "…"
    return text


def retrieve_saved_context(query: str, *, top_k: int = 6) -> dict:
    """Retrieve grounded context drawn only from the reader's saved papers.

    Returns ``{"sources": [{paper_id, title, score, snippet}], "context": str,
    "no_saved_papers": bool}``. When the reader has saved nothing, ``sources`` is
    empty, ``context`` is "" and ``no_saved_papers`` is True.
    """
    saved_ids = _saved_paper_ids()
    if not saved_ids:
        return {"sources": [], "context": "", "no_saved_papers": True}

    ranked = search_hybrid(query, top_k=max(top_k * 4, top_k))
    ordered_ids: list[int] = [r["paper_id"] for r in ranked if r["paper_id"] in saved_ids]
    score_by_id: dict[int, float] = {r["paper_id"]: r.get("rrf_score", 0.0) for r in ranked}

    # If hybrid search returned nothing useful (e.g. empty index in tests), fall
    # back to the saved papers themselves so the reader still gets context.
    if not ordered_ids:
        ordered_ids = sorted(saved_ids)

    selected_ids = ordered_ids[:top_k]
    papers_by_id = {p.id: p for p in Paper.query.filter(Paper.id.in_(selected_ids)).all()}

    sources: list[dict] = []
    blocks: list[str] = []
    for pid in selected_ids:
        paper = papers_by_id.get(pid)
        if paper is None:
            continue
        sources.append(
            {
                "paper_id": paper.id,
                "title": paper.title,
                "score": round(float(score_by_id.get(pid, 0.0)), 6),
                "snippet": _snippet(paper),
            }
        )
        blocks.append(_build_paper_block(paper, query))

    return {"sources": sources, "context": "\n\n---\n\n".join(blocks), "no_saved_papers": False}


def _build_client(app=None):
    """Reuse the scrape pipeline's LLM-client builder so config/key handling can't drift."""
    application = app or current_app
    from app.services.scrape_engine import _create_llm_client

    client, _interests = _create_llm_client(application)
    return client


def _synthesize(client, query: str, context: str) -> str | None:
    """Call the low-level completion helper; return text or None on any failure."""
    user_prompt = f"Context from saved papers:\n\n{context}\n\nQuestion: {query}"
    try:
        # Use the throttled public wrapper so chat respects the LLM concurrency cap
        # instead of bypassing the semaphore via the private _create_completion.
        response = client.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=600,
            temperature=0.2,
        )
        content = response.choices[0].message.content
    except Exception:  # noqa: BLE001 — mirror llm_client's None-on-failure convention
        return None
    return content.strip() if isinstance(content, str) and content.strip() else None


def answer_query(query: str, *, top_k: int = 6, app=None) -> dict:
    """Answer ``query`` from the saved corpus, synthesizing with the LLM if available.

    Returns ``{"query", "synthesis", "llm_used", "sources", "no_saved_papers"}``.
    ``synthesis`` is None whenever the LLM is disabled, no key is available, the
    completion fails, or there is no context to ground an answer in.
    """
    retrieval = retrieve_saved_context(query, top_k=top_k)
    sources = retrieval["sources"]
    no_saved_papers = retrieval["no_saved_papers"]

    synthesis: str | None = None
    llm_used = False

    if not no_saved_papers and retrieval["context"]:
        client = _build_client(app=app)
        if client is not None:
            synthesis = _synthesize(client, query, retrieval["context"])
            llm_used = synthesis is not None

    return {
        "query": query,
        "synthesis": synthesis,
        "llm_used": llm_used,
        "sources": sources,
        "no_saved_papers": no_saved_papers,
    }
