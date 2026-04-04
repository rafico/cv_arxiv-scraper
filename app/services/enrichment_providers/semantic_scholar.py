"""Semantic Scholar enrichment provider."""

from __future__ import annotations

import logging
from typing import Any

from app.services.enrichment_providers.base import (
    DEFAULT_CACHE_TTL_HOURS,
    EnrichmentProvider,
    get_cached_payloads,
    store_cached_payloads,
)

LOGGER = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"


class SemanticScholarProvider(EnrichmentProvider):
    source = "semantic_scholar"

    def __init__(self, *, ttl_hours: int = DEFAULT_CACHE_TTL_HOURS, request_fn=None) -> None:
        self.ttl_hours = ttl_hours
        self._request_fn = request_fn

    def fetch_batch(self, arxiv_ids: list[str], session=None) -> dict[str, dict[str, Any]]:
        from app.services.http_client import request_with_backoff

        if not arxiv_ids:
            return {}

        request_fn = self._request_fn or request_with_backoff
        cached, missing_ids, paper_by_arxiv_id = get_cached_payloads(arxiv_ids, source=self.source)
        if not missing_ids:
            return cached

        payload = {"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in missing_ids]}
        params = {"fields": "citationCount,influentialCitationCount,paperId"}

        try:
            response = request_fn(
                "POST",
                SEMANTIC_SCHOLAR_BATCH_URL,
                json=payload,
                params=params,
                session=session,
                timeout=15,
            )
            if not response:
                return cached

            data = response.json()
            fetched: dict[str, dict[str, Any]] = {}
            for idx, item in enumerate(data):
                if item is None:
                    continue
                arxiv_id = missing_ids[idx]
                fetched[arxiv_id] = {
                    "citation_count": item.get("citationCount"),
                    "influential_citation_count": item.get("influentialCitationCount"),
                    "semantic_scholar_id": item.get("paperId"),
                }

            store_cached_payloads(
                fetched,
                source=self.source,
                paper_by_arxiv_id=paper_by_arxiv_id,
                ttl_hours=self.ttl_hours,
            )
            return {**cached, **fetched}
        except Exception as exc:
            LOGGER.warning("Failed to fetch citations from Semantic Scholar: %s", exc)
            return cached
