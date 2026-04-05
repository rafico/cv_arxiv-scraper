"""Hybrid search combining BM25 (FTS5) and semantic (FAISS) retrieval."""

from __future__ import annotations

import logging

from sqlalchemy import text

from app.models import db

LOGGER = logging.getLogger(__name__)

RRF_K = 60  # Reciprocal Rank Fusion constant


def _sanitize_fts5_query(query: str) -> str:
    """Escape user input for safe use in FTS5 MATCH by wrapping tokens in double quotes."""
    # Remove existing double quotes and wrap each token as a quoted phrase
    # to prevent FTS5 operator injection (NEAR, OR, NOT, *, etc.)
    cleaned = query.replace('"', "")
    tokens = cleaned.split()
    if not tokens:
        return '""'
    return " ".join(f'"{token}"' for token in tokens)


def _fts5_available() -> bool:
    """Check if the papers_fts table exists."""
    try:
        db.session.execute(text("SELECT COUNT(*) FROM papers_fts LIMIT 1"))
        return True
    except Exception:
        db.session.rollback()
        return False


def search_bm25(query: str, limit: int = 50) -> list[tuple[int, float]]:
    """Full-text search using SQLite FTS5. Returns [(paper_id, bm25_score)]."""
    if not query.strip():
        return []

    if not _fts5_available():
        return []

    try:
        safe_query = _sanitize_fts5_query(query)
        rows = db.session.execute(
            text("SELECT rowid, rank FROM papers_fts WHERE papers_fts MATCH :query ORDER BY rank LIMIT :limit"),
            {"query": safe_query, "limit": limit},
        ).fetchall()
        # FTS5 rank is negative (more negative = better match), convert to positive score
        return [(int(row[0]), -float(row[1])) for row in rows]
    except Exception as exc:
        LOGGER.warning("FTS5 search failed: %s", exc)
        db.session.rollback()
        return []


def search_semantic(query: str, top_k: int = 50) -> list[tuple[int, float]]:
    """Semantic search using FAISS embeddings. Returns [(paper_id, cosine_score)]."""
    if not query.strip():
        return []

    try:
        from app.services.embeddings import get_embedding_service

        service = get_embedding_service()
        if service.index_count() == 0:
            return []
        return service.search(query, top_k=top_k)
    except Exception as exc:
        LOGGER.warning("Semantic search failed: %s", exc)
        return []


def search_hybrid(
    query: str,
    *,
    top_k: int = 30,
    bm25_weight: float = 0.4,
    semantic_weight: float = 0.6,
) -> list[dict]:
    """
    Combine BM25 and semantic search via Reciprocal Rank Fusion.

    Returns [{paper_id, rrf_score, bm25_rank, semantic_rank}] sorted by rrf_score desc.
    """
    if not query.strip():
        return []

    bm25_results = search_bm25(query, limit=top_k * 2)
    semantic_results = search_semantic(query, top_k=top_k * 2)

    # If only one system has results, use that
    if not bm25_results and not semantic_results:
        return []

    # Build rank maps (1-indexed)
    bm25_ranks: dict[int, int] = {}
    for rank, (pid, _score) in enumerate(bm25_results, 1):
        bm25_ranks[pid] = rank

    semantic_ranks: dict[int, int] = {}
    for rank, (pid, _score) in enumerate(semantic_results, 1):
        semantic_ranks[pid] = rank

    # Collect all candidate paper IDs
    all_pids = set(bm25_ranks.keys()) | set(semantic_ranks.keys())

    # Compute RRF scores
    scored = []
    for pid in all_pids:
        rrf_score = 0.0
        bm25_rank = bm25_ranks.get(pid)
        semantic_rank = semantic_ranks.get(pid)

        if bm25_rank is not None:
            rrf_score += bm25_weight / (RRF_K + bm25_rank)
        if semantic_rank is not None:
            rrf_score += semantic_weight / (RRF_K + semantic_rank)

        scored.append(
            {
                "paper_id": pid,
                "rrf_score": round(rrf_score, 6),
                "bm25_rank": bm25_rank,
                "semantic_rank": semantic_rank,
            }
        )

    scored.sort(key=lambda x: x["rrf_score"], reverse=True)
    return scored[:top_k]
