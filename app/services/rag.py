"""Conversational RAG over the reader's papers.

Retrieval scope, narrowest first: an explicit ``paper_ids`` set (a collection),
else the papers the reader saved (``PaperFeedback`` rows with ``action ==
"save"``), else the whole corpus via hybrid search. The response labels which
scope answered (``"collection"|"saved"|"corpus"``) because corpus answers mix in
unvetted papers. Scoped retrieval ranks every member exactly by embedding
similarity, so no member is lost to a global top-k cut-off.

Synthesis is optional: when the LLM is disabled or no API key is available we
return the numbered sources with ``synthesis=None`` and ``llm_used=False`` — the
list itself answers "which of my papers discuss X, and where". The network is
only touched when a client is successfully built, and any failure degrades back
to ``synthesis=None``.
"""

from __future__ import annotations

import logging

from flask import current_app

from app.enums import FeedbackAction
from app.models import Paper, PaperFeedback, db
from app.services.search import search_hybrid

LOGGER = logging.getLogger(__name__)

# Per-paper character budgets for the context block. Keeps the prompt (and the
# extractive payload returned when synthesis is off) within a sane size.
_ABSTRACT_CHAR_BUDGET = 250
_EXCERPT_CHAR_BUDGET = 900
_SNIPPET_CHAR_BUDGET = 320
# The abstract has its own "Abstract:" line; the excerpt comes from the body.
_SKIP_SECTIONS = {"abstract", "references", "acknowledgments"}

_SYSTEM_PROMPT = (
    "You are a research assistant answering questions about a reader's papers. "
    "Answer ONLY using the numbered context papers. After each claim, cite the supporting "
    "paper number in square brackets, e.g. [1] or [2][3], using only numbers that exist in the context. "
    "If the context does not contain enough information to answer, say so plainly rather than guessing."
)


def _saved_paper_ids() -> set[int]:
    """Return the ids of papers the reader has saved."""
    rows = db.session.query(PaperFeedback.paper_id).filter(PaperFeedback.action == FeedbackAction.SAVE.value).all()
    return {row[0] for row in rows}


def _rank_scope(query: str, scope: set[int]) -> list[tuple[int, float]]:
    """Rank every paper in ``scope`` by exact cosine similarity to ``query``.

    Members without an embedding (or all of them, when the model is unavailable)
    trail in id order with score 0 so the reader still gets context.
    """
    ranked: list[tuple[int, float]] = []
    try:
        from app.services.embeddings import get_embedding_service

        service = get_embedding_service()
        ids, vectors = service.get_paper_vectors(sorted(scope))
        if ids:
            scores = (vectors @ service.encode([query])[0]).tolist()
            ranked = sorted(zip(ids, scores), key=lambda r: r[1], reverse=True)
    except Exception:  # noqa: BLE001 — ranking is best-effort; degrade to id order
        LOGGER.debug("Scoped ranking unavailable; falling back to id order", exc_info=True)
    ranked_ids = {pid for pid, _ in ranked}
    # ponytail: unembedded members trail unranked; BM25-rank them if embed backfill lags in practice.
    return ranked + [(pid, 0.0) for pid in sorted(scope) if pid not in ranked_ids]


def _build_paper_block(n: int, paper: Paper, query: str) -> tuple[str, str | None]:
    """Return ``([n] title + short abstract + best body excerpt, excerpt section_type)``."""
    from app.services.paper_chat import _paper_chunks, _rank_chunks, _truncate  # local: paper_chat imports rag

    parts = [f"[{n}] {paper.title}"]
    abstract = (paper.abstract_text or paper.summary_text or "").strip()
    if abstract:
        parts.append(f"Abstract: {_truncate(abstract, _ABSTRACT_CHAR_BUDGET)}")
    body = [c for c in _paper_chunks(paper) if c["section_type"] not in _SKIP_SECTIONS]
    best = _rank_chunks(query, body, top_k=1)
    if not best:
        return "\n".join(parts), None
    parts.append(f"Excerpt ({best[0]['section_type']}): {_truncate(best[0]['text'], _EXCERPT_CHAR_BUDGET)}")
    return "\n".join(parts), best[0]["section_type"]


def _snippet(paper: Paper) -> str:
    text = (paper.abstract_text or paper.summary_text or "").strip()
    if len(text) > _SNIPPET_CHAR_BUDGET:
        return text[: _SNIPPET_CHAR_BUDGET - 1].rstrip() + "…"
    return text


def retrieve_saved_context(query: str, *, top_k: int = 6, paper_ids: list[int] | None = None) -> dict:
    """Retrieve numbered, grounded context for ``query`` (scope rules in the module docstring).

    Returns ``{"sources": [{n, paper_id, title, score, snippet, section}],
    "context": str, "scope": "collection"|"saved"|"corpus"}``; ``section`` names
    where the best-matching passage sits (None for abstract-only papers).
    """
    if paper_ids is not None:
        scope, ids = "collection", set(paper_ids)
    else:
        ids = _saved_paper_ids()
        scope = "saved" if ids else "corpus"

    if scope == "corpus":
        # Over-fetch: skipped hits are dropped below.
        ranked = [(r["paper_id"], r.get("rrf_score", 0.0)) for r in search_hybrid(query, top_k=top_k * 2)]
    else:
        ranked = _rank_scope(query, ids)

    # Skipped (hidden) papers never become sources, as in the collection view, .bib and MCP.
    # ponytail: loads every scoped row; filter ids first if collections reach thousands.
    visible = Paper.query.filter(Paper.id.in_([pid for pid, _ in ranked]), Paper.is_hidden.is_(False))
    papers_by_id = {p.id: p for p in visible.all()}

    sources: list[dict] = []
    blocks: list[str] = []
    for pid, score in ranked:
        paper = papers_by_id.get(pid)
        if paper is None:
            continue
        if len(sources) == top_k:
            break
        n = len(sources) + 1
        block, section = _build_paper_block(n, paper, query)
        sources.append(
            {
                "n": n,
                "paper_id": paper.id,
                "title": paper.title,
                "score": round(float(score), 6),
                "snippet": _snippet(paper),
                "section": section,
            }
        )
        blocks.append(block)

    return {"sources": sources, "context": "\n\n---\n\n".join(blocks), "scope": scope}


def _build_client(app=None):
    """Reuse the scrape pipeline's LLM-client builder so config/key handling can't drift."""
    application = app or current_app
    from app.services.scrape_engine import _create_llm_client

    client, _interests = _create_llm_client(application)
    return client


def build_llm_client(app=None):
    """Public alias of the shared chat LLM-client builder (also used by paper_chat)."""
    return _build_client(app=app)


def _synthesize(client, query: str, context: str) -> str | None:
    """Call the low-level completion helper; return text or None on any failure."""
    user_prompt = f"Context papers:\n\n{context}\n\nQuestion: {query}"
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


def answer_query(query: str, *, top_k: int = 6, paper_ids: list[int] | None = None, app=None) -> dict:
    """Answer ``query`` from the scoped papers, synthesizing with the LLM if available.

    Returns ``{"query", "synthesis", "llm_used", "sources", "scope", "verifications"}``.
    ``synthesis`` is None whenever the LLM is disabled, no key is available, the
    completion fails, or there is no context to ground an answer in; otherwise
    any ``[n]`` marker that maps to no source is stripped.
    """
    retrieval = retrieve_saved_context(query, top_k=top_k, paper_ids=paper_ids)
    sources = retrieval["sources"]

    synthesis: str | None = None
    llm_used = False

    if retrieval["context"]:
        client = _build_client(app=app)
        if client is not None:
            synthesis = _synthesize(client, query, retrieval["context"])
            llm_used = synthesis is not None

    # Strip fabricated [n] markers, then resolve any arXiv ids / DOIs / paper titles
    # the synthesis names against the local corpus (None when no synthesis).
    verifications = None
    if synthesis is not None:
        from app.services import citation_verifier
        from app.services.paper_chat import _ground_answer  # local: paper_chat imports rag

        synthesis, _cited, _stripped = _ground_answer(synthesis, len(sources))
        verifications = citation_verifier.verify_text(synthesis)

    return {
        "query": query,
        "synthesis": synthesis,
        "llm_used": llm_used,
        "sources": sources,
        "scope": retrieval["scope"],
        "verifications": verifications,
    }
