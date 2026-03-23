"""Semantic Scholar citations integration."""

from __future__ import annotations

import logging
from typing import Any

from app.services.http_client import request_with_backoff

LOGGER = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

def fetch_citations_batch(arxiv_ids: list[str], session=None) -> dict[str, dict[str, Any]]:
    """
    Fetch citation data from Semantic Scholar for a batch of arXiv IDs.
    Returns a dict mapping arXiv ID to citation data dict.
    """
    if not arxiv_ids:
        return {}
        
    payload = {
        "ids": [f"ARXIV:{arxiv_id}" for arxiv_id in arxiv_ids]
    }
    params = {
        "fields": "citationCount,influentialCitationCount,paperId"
    }

    try:
        response = request_with_backoff(
            "POST",
            SEMANTIC_SCHOLAR_BATCH_URL,
            json=payload,
            params=params,
            session=session,
            timeout=15,
        )
        if not response:
            return {}
            
        data = response.json()
        results = {}
        for idx, item in enumerate(data):
            if item is None:
                continue
            arxiv_id = arxiv_ids[idx]
            results[arxiv_id] = {
                "citation_count": item.get("citationCount"),
                "influential_citation_count": item.get("influentialCitationCount"),
                "semantic_scholar_id": item.get("paperId"),
            }
        return results
    except Exception as exc:
        LOGGER.warning("Failed to fetch citations from Semantic Scholar: %s", exc)
        return {}
