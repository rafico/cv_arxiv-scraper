"""Shared interfaces and cache helpers for enrichment providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from flask import has_app_context

from app.services.text import now_utc

DEFAULT_CACHE_TTL_HOURS = 168


class EnrichmentProvider(Protocol):
    source: str

    def fetch_batch(self, arxiv_ids: list[str], **kwargs: Any) -> dict[str, dict[str, Any]]: ...


def _ordered_arxiv_ids(arxiv_ids: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for arxiv_id in arxiv_ids:
        if not arxiv_id or arxiv_id in seen:
            continue
        ordered.append(arxiv_id)
        seen.add(arxiv_id)
    return ordered


def get_cached_payloads(
    arxiv_ids: Sequence[str],
    *,
    source: str,
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    """Return cached payloads keyed by arXiv id, plus missing ids and paper mapping."""
    ordered_ids = _ordered_arxiv_ids(arxiv_ids)
    if not ordered_ids or not has_app_context():
        return {}, ordered_ids, {}

    from app.models import EnrichmentCache, Paper

    papers = Paper.query.filter(Paper.arxiv_id.in_(ordered_ids)).all()
    paper_by_arxiv_id = {paper.arxiv_id: paper for paper in papers if paper.arxiv_id}
    if not paper_by_arxiv_id:
        return {}, ordered_ids, {}

    cache_rows = EnrichmentCache.query.filter(
        EnrichmentCache.source == source,
        EnrichmentCache.paper_id.in_([paper.id for paper in paper_by_arxiv_id.values()]),
    ).all()
    cache_by_paper_id = {row.paper_id: row for row in cache_rows}
    reference_time = now_utc()

    cached: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for arxiv_id in ordered_ids:
        paper = paper_by_arxiv_id.get(arxiv_id)
        cache_row = cache_by_paper_id.get(paper.id) if paper is not None else None
        if cache_row is not None and cache_row.is_fresh(reference_time=reference_time):
            cached[arxiv_id] = dict(cache_row.data or {})
        else:
            missing.append(arxiv_id)

    return cached, missing, paper_by_arxiv_id


def store_cached_payloads(
    payloads: Mapping[str, dict[str, Any]],
    *,
    source: str,
    paper_by_arxiv_id: Mapping[str, Any],
    ttl_hours: int = DEFAULT_CACHE_TTL_HOURS,
) -> None:
    """Persist provider payloads for papers already stored in the DB."""
    if not payloads or not paper_by_arxiv_id or not has_app_context():
        return

    from app.models import EnrichmentCache, db

    paper_ids = [paper.id for arxiv_id, paper in paper_by_arxiv_id.items() if arxiv_id in payloads]
    if not paper_ids:
        return

    existing_rows = EnrichmentCache.query.filter(
        EnrichmentCache.source == source,
        EnrichmentCache.paper_id.in_(paper_ids),
    ).all()
    existing_by_paper_id = {row.paper_id: row for row in existing_rows}
    fetched_at = now_utc()

    for arxiv_id, payload in payloads.items():
        paper = paper_by_arxiv_id.get(arxiv_id)
        if paper is None:
            continue

        cache_row = existing_by_paper_id.get(paper.id)
        if cache_row is None:
            cache_row = EnrichmentCache(paper_id=paper.id, source=source)
            db.session.add(cache_row)
            existing_by_paper_id[paper.id] = cache_row

        cache_row.data = dict(payload)
        cache_row.fetched_at = fetched_at
        cache_row.ttl_hours = ttl_hours

    db.session.commit()
