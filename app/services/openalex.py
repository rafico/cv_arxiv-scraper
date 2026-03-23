"""OpenAlex enrichment — topics, OA status, and citation data."""

from __future__ import annotations

import logging
from typing import Any

from app.services.http_client import request_with_backoff

LOGGER = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def _parse_openalex_work(work: dict) -> dict[str, Any]:
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
    if not arxiv_ids:
        return {}

    results: dict[str, dict[str, Any]] = {}

    # OpenAlex supports pipe-separated filter values, up to ~50 per request
    batch_size = 50
    for i in range(0, len(arxiv_ids), batch_size):
        batch = arxiv_ids[i : i + batch_size]
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
            response = request_with_backoff(
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
                # Extract arXiv ID from DOI: https://doi.org/10.48550/arxiv.XXXX.XXXXX
                for aid in batch:
                    if aid.lower() in doi:
                        results[aid] = _parse_openalex_work(work)
                        break

        except Exception as exc:
            LOGGER.warning("Failed to fetch OpenAlex data for batch: %s", exc)

    return results
