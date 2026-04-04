"""OpenAlex enrichment provider."""

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

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def parse_openalex_work(work: dict) -> dict[str, Any]:
    """Extract relevant fields from an OpenAlex work object."""
    topics = []
    for topic in work.get("topics", []):
        name = topic.get("display_name")
        score = topic.get("score")
        if name:
            topics.append({"name": name, "score": score})

    oa_info = work.get("open_access", {}) or {}

    return {
        "openalex_id": work.get("id", "").replace("https://openalex.org/", ""),
        "openalex_topics": topics,
        "oa_status": oa_info.get("oa_status"),
        "openalex_cited_by_count": work.get("cited_by_count"),
        "referenced_works_count": len(work.get("referenced_works", [])),
    }


class OpenAlexProvider(EnrichmentProvider):
    source = "openalex"

    def __init__(self, *, ttl_hours: int = DEFAULT_CACHE_TTL_HOURS, request_fn=None) -> None:
        self.ttl_hours = ttl_hours
        self._request_fn = request_fn

    def fetch_batch(
        self,
        arxiv_ids: list[str],
        session=None,
        email: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        from app.services.http_client import request_with_backoff

        if not arxiv_ids:
            return {}

        request_fn = self._request_fn or request_with_backoff
        cached, missing_ids, paper_by_arxiv_id = get_cached_payloads(arxiv_ids, source=self.source)
        if not missing_ids:
            return cached

        fetched: dict[str, dict[str, Any]] = {}

        batch_size = 50
        for i in range(0, len(missing_ids), batch_size):
            batch = missing_ids[i : i + batch_size]
            dois = [f"https://doi.org/10.48550/arXiv.{aid}" for aid in batch]
            doi_filter = "|".join(dois)

            params: dict[str, str] = {
                "filter": f"doi:{doi_filter}",
                "select": "id,doi,open_access,cited_by_count,referenced_works,topics",
                "per_page": str(len(batch)),
            }
            if email:
                params["mailto"] = email

            try:
                response = request_fn(
                    "GET",
                    OPENALEX_WORKS_URL,
                    params=params,
                    session=session,
                    timeout=15,
                )
                if not response:
                    continue

                data = response.json()
                for work in data.get("results", []):
                    doi = (work.get("doi") or "").lower()
                    for aid in batch:
                        if aid.lower() in doi:
                            fetched[aid] = parse_openalex_work(work)
                            break
            except Exception as exc:
                LOGGER.warning("Failed to fetch OpenAlex data for batch: %s", exc)

        store_cached_payloads(
            fetched,
            source=self.source,
            paper_by_arxiv_id=paper_by_arxiv_id,
            ttl_hours=self.ttl_hours,
        )
        return {**cached, **fetched}
