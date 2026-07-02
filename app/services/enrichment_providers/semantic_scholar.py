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

# The Semantic Scholar batch endpoint caps each request at 500 ids; larger
# payloads error (400/413). Chunk to stay under the cap.
SEMANTIC_SCHOLAR_BATCH_LIMIT = 500


class SemanticScholarProvider(EnrichmentProvider):
    source = "semantic_scholar"

    def __init__(self, *, ttl_hours: int = DEFAULT_CACHE_TTL_HOURS, request_fn=None, api_key: str | None = None) -> None:
        self.ttl_hours = ttl_hours
        self._request_fn = request_fn
        self._api_key = api_key

    def fetch_batch(self, arxiv_ids: list[str], session=None) -> dict[str, dict[str, Any]]:  # type: ignore[override]  # provider-specific kwargs; base Protocol uses **kwargs
        from app.services.http_client import request_with_backoff
        from app.services.secret_files import resolve_data_source_key

        if not arxiv_ids:
            return {}

        request_fn = self._request_fn or request_with_backoff
        cached, missing_ids, paper_by_arxiv_id = get_cached_payloads(arxiv_ids, source=self.source)
        if not missing_ids:
            return cached

        params = {"fields": "citationCount,influentialCitationCount,paperId"}
        # Extra kwargs are only passed when they carry a value: injected request_fn
        # doubles (tests) keep the narrow legacy signature and must not receive them.
        extra_kwargs: dict[str, Any] = {}
        api_key = self._api_key or resolve_data_source_key("semantic_scholar")
        if api_key:
            extra_kwargs["headers"] = {"x-api-key": api_key}
        if request_fn is request_with_backoff:
            # Semantic Scholar issues new keys at ~1 request/second; the shared
            # "bulk" profile (≤1 req / 3 s) paces safely under that, and the batch
            # endpoint (500 ids per POST) keeps total request counts tiny.
            extra_kwargs["rate_limit_profile"] = "bulk"
        fetched: dict[str, dict[str, Any]] = {}

        for i in range(0, len(missing_ids), SEMANTIC_SCHOLAR_BATCH_LIMIT):
            batch = missing_ids[i : i + SEMANTIC_SCHOLAR_BATCH_LIMIT]
            payload = {"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in batch]}

            try:
                response = request_fn(
                    "POST",
                    SEMANTIC_SCHOLAR_BATCH_URL,
                    json=payload,
                    params=params,
                    session=session,
                    timeout=15,
                    **extra_kwargs,
                )
                # The real request_with_backoff raises on failure and always
                # returns a truthy Response, so this guard is dead on that path.
                # It is retained deliberately to tolerate an injected request_fn
                # (test doubles) that returns a falsy/None response instead of
                # raising.
                if not response:
                    continue

                data = response.json()
                for idx, item in enumerate(data):
                    if item is None:
                        continue
                    # Map by position WITHIN the current chunk; missing_ids[idx]
                    # would misattribute every chunk after the first.
                    arxiv_id = batch[idx]
                    fetched[arxiv_id] = {
                        "citation_count": item.get("citationCount"),
                        "influential_citation_count": item.get("influentialCitationCount"),
                        "semantic_scholar_id": item.get("paperId"),
                    }
            except Exception as exc:
                # One failed chunk must not abandon the rest of the batch.
                LOGGER.warning("Failed to fetch citations from Semantic Scholar: %s", exc)

        store_cached_payloads(
            fetched,
            source=self.source,
            paper_by_arxiv_id=paper_by_arxiv_id,
            ttl_hours=self.ttl_hours,
        )
        return {**cached, **fetched}
