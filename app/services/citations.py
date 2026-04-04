"""Semantic Scholar citations integration."""

from __future__ import annotations

from typing import Any

from app.services.enrichment_providers import SemanticScholarProvider
from app.services.http_client import request_with_backoff


def fetch_citations_batch(arxiv_ids: list[str], session=None) -> dict[str, dict[str, Any]]:
    """
    Fetch citation data from Semantic Scholar for a batch of arXiv IDs.
    Returns a dict mapping arXiv ID to citation data dict.
    """
    provider = SemanticScholarProvider(request_fn=request_with_backoff)
    return provider.fetch_batch(arxiv_ids, session=session)
