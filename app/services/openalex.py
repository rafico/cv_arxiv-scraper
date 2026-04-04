"""OpenAlex enrichment — topics, OA status, and citation data."""

from __future__ import annotations

from typing import Any

from app.services.enrichment_providers import OpenAlexProvider, parse_openalex_work
from app.services.http_client import request_with_backoff

_parse_openalex_work = parse_openalex_work


def fetch_openalex_batch(
    arxiv_ids: list[str],
    session=None,
    email: str | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Fetch OpenAlex metadata for papers by arXiv ID.

    Looks up papers via their arXiv DOI (10.48550/arXiv.{id}).
    Returns {arxiv_id: {openalex_id, openalex_topics, oa_status, ...}}.
    """
    provider = OpenAlexProvider(request_fn=request_with_backoff)
    return provider.fetch_batch(arxiv_ids, session=session, email=email)
