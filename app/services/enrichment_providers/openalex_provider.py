"""OpenAlex enrichment provider."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.services.enrichment_providers.base import (
    DEFAULT_CACHE_TTL_HOURS,
    EnrichmentProvider,
    get_cached_payloads,
    store_cached_payloads,
)

LOGGER = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

# OpenAlex made API keys mandatory (~Feb 2026); the free tier authenticates via an
# ``api_key`` query parameter (per docs.openalex.org → developers.openalex.org).
# Keyless requests get throttled/rejected with these statuses.
_AUTH_FAILURE_STATUSES = frozenset({401, 403, 429})

# Emit the "you need a key now" hint once per process, not once per batch — a
# 500-paper scrape would otherwise repeat it ten times.
_missing_key_warned = False


def _warn_missing_key_once() -> None:
    global _missing_key_warned
    if _missing_key_warned:
        return
    _missing_key_warned = True
    LOGGER.warning(
        "OpenAlex now requires an API key — set one in Settings → Automation → Data Sources "
        "(stored in .openalex_api_key) or via the OPENALEX_API_KEY env var. "
        "Continuing without OpenAlex enrichment."
    )


def parse_openalex_work(work: dict) -> dict[str, Any]:
    """Extract relevant fields from an OpenAlex work object.

    Null-safe: OpenAlex returns explicit ``null`` for absent fields, so ``work.get``
    with a default still yields None. Coerce with ``or`` so a single sparse work
    can't crash (and abandon) the rest of the batch.
    """
    topics = []
    for topic in work.get("topics") or []:
        name = topic.get("display_name")
        score = topic.get("score")
        if name:
            topics.append({"name": name, "score": score})

    oa_info = work.get("open_access") or {}

    return {
        "openalex_id": (work.get("id") or "").replace("https://openalex.org/", ""),
        "openalex_topics": topics,
        "oa_status": oa_info.get("oa_status"),
        "openalex_cited_by_count": work.get("cited_by_count"),
        "referenced_works_count": len(work.get("referenced_works") or []),
    }


class OpenAlexProvider(EnrichmentProvider):
    source = "openalex"

    def __init__(self, *, ttl_hours: int = DEFAULT_CACHE_TTL_HOURS, request_fn=None, api_key: str | None = None) -> None:
        self.ttl_hours = ttl_hours
        self._request_fn = request_fn
        self._api_key = api_key

    def fetch_batch(  # type: ignore[override]  # provider-specific kwargs; base Protocol uses **kwargs
        self,
        arxiv_ids: list[str],
        session=None,
        email: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        from app.services.http_client import request_with_backoff
        from app.services.secret_files import resolve_data_source_key

        if not arxiv_ids:
            return {}

        request_fn = self._request_fn or request_with_backoff
        api_key = self._api_key or resolve_data_source_key("openalex")
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
            if api_key:
                params["api_key"] = api_key

            try:
                response = request_fn(
                    "GET",
                    OPENALEX_WORKS_URL,
                    params=params,
                    session=session,
                    timeout=15,
                )
                # The real request_with_backoff raises on failure and always
                # returns a truthy Response, so this guard is dead on that path.
                # It is retained deliberately to tolerate an injected request_fn
                # (test doubles / the fetch_openalex_batch shim) that returns a
                # falsy/None response instead of raising.
                if not response:
                    continue

                data = response.json()
                batch_by_id = {aid.lower(): aid for aid in batch}
                for work in data.get("results", []):
                    try:
                        doi = (work.get("doi") or "").lower()
                        if "arxiv." not in doi:
                            continue
                        # Map the work back by its *exact* arXiv id, not a substring:
                        # `aid in doi` wrongly attributes ".../arxiv.2301.00012" to the
                        # shorter id "2301.0001". Strip any trailing version (vN).
                        work_id = re.sub(r"v\d+$", "", doi.rsplit("arxiv.", 1)[-1])
                        aid = batch_by_id.get(work_id)
                        if aid is not None:
                            fetched[aid] = parse_openalex_work(work)
                    except Exception as exc:
                        # One malformed work must not abandon the rest of the batch.
                        LOGGER.warning("Skipping malformed OpenAlex work: %s", exc)
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in _AUTH_FAILURE_STATUSES and not api_key:
                    _warn_missing_key_once()
                LOGGER.warning("Failed to fetch OpenAlex data for batch: %s", exc)

        store_cached_payloads(
            fetched,
            source=self.source,
            paper_by_arxiv_id=paper_by_arxiv_id,
            ttl_hours=self.ttl_hours,
        )
        return {**cached, **fetched}
