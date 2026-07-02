"""Hugging Face Papers enrichment provider (community buzz + code/project links).

Papers with Code shut down in July 2025; huggingface.co/papers is its successor
as the community code-link and buzz source. The API is keyless and keyed by
arXiv id. Most papers are never submitted there, so a 404 is the normal "not
featured" case: it is cached as an empty payload (a durable miss), not treated
as an error.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from app.services.enrichment_providers.base import (
    EnrichmentProvider,
    get_cached_payloads,
    store_cached_payloads,
)

LOGGER = logging.getLogger(__name__)

HF_PAPER_API_URL = "https://huggingface.co/api/papers/{arxiv_id}"
# Upvotes/comments keep accruing for days after a paper hits the daily page, and
# papers are often submitted days after publication — refresh hits (and retry
# 404 misses) faster than the 7-day default TTL.
HUGGINGFACE_CACHE_TTL_HOURS = 72
# The endpoint is per-paper (no batch form); bound one run's request count.
DEFAULT_MAX_FETCHES_PER_RUN = 200


def parse_hf_paper(data: dict) -> dict[str, Any]:
    """Extract relevant fields from a Hugging Face paper object.

    ``comments`` is a list only when the request asked for ``?field=comments``;
    the daily_papers listing carries ``numComments`` instead, so accept both.
    """
    comments = data.get("comments")
    comments_count = len(comments) if isinstance(comments, list) else data.get("numComments")
    return {
        "hf_upvotes": data.get("upvotes"),
        "hf_comments_count": comments_count,
        "github_repo_url": data.get("githubRepo") or None,
        "project_page_url": data.get("projectPage") or None,
    }


def huggingface_resource_links(payload: dict | None) -> list[dict[str, str]]:
    """Typed resource-link dicts for a parsed HF payload (merge_resource_links shape)."""
    links: list[dict[str, str]] = []
    if not payload:
        return links
    if payload.get("github_repo_url"):
        links.append({"type": "code", "label": "Code", "url": payload["github_repo_url"]})
    if payload.get("project_page_url"):
        links.append({"type": "project", "label": "Project", "url": payload["project_page_url"]})
    return links


def _is_rate_limited(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return response is not None and response.status_code in (403, 429)


def _is_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return response is not None and response.status_code == 404


class HuggingFaceProvider(EnrichmentProvider):
    source = "huggingface"

    def __init__(
        self,
        *,
        ttl_hours: int = HUGGINGFACE_CACHE_TTL_HOURS,
        request_fn=None,
        max_fetches: int = DEFAULT_MAX_FETCHES_PER_RUN,
    ) -> None:
        self.ttl_hours = ttl_hours
        self._request_fn = request_fn
        self.max_fetches = max_fetches
        # Set when a batch stops early due to rate limiting, so a caller (e.g.
        # the CLI backfill) can stop advancing its cursor past unfetched papers.
        self.rate_limited = False

    def fetch_batch(  # type: ignore[override]  # provider-specific kwargs; base Protocol uses **kwargs
        self,
        arxiv_ids: list[str],
        session: requests.Session | None = None,
    ) -> dict[str, dict[str, Any]]:
        from app.services.http_client import request_with_backoff

        if not arxiv_ids:
            return {}

        request_fn = self._request_fn or request_with_backoff
        cached, missing_ids, paper_by_arxiv_id = get_cached_payloads(arxiv_ids, source=self.source)

        fetched: dict[str, dict[str, Any]] = {}
        fetches = 0
        for arxiv_id in missing_ids:
            if fetches >= self.max_fetches:
                LOGGER.info("Hugging Face fetch cap (%d) reached; deferring remaining papers", self.max_fetches)
                break
            fetches += 1

            try:
                response = request_fn(
                    "GET",
                    HF_PAPER_API_URL.format(arxiv_id=arxiv_id),
                    params={"field": "comments"},
                    session=session,
                    timeout=15,
                    attempts=2,
                    rate_limit_profile="bulk",
                )
            except Exception as exc:
                if _is_not_found(exc):
                    # Normal case: the paper was never submitted to HF. Cache the
                    # miss so it is not re-queried every run until the TTL lapses.
                    fetched[arxiv_id] = {}
                    continue
                if _is_rate_limited(exc):
                    LOGGER.warning("Hugging Face API rate limited; skipping remaining papers: %s", exc)
                    self.rate_limited = True
                    break
                LOGGER.warning("Hugging Face fetch failed for %s: %s", arxiv_id, exc)
                continue

            try:
                fetched[arxiv_id] = parse_hf_paper(response.json())
            except Exception as exc:
                # A non-JSON 200 must skip this one paper, not abort the whole run.
                LOGGER.warning("Hugging Face payload parse failed for %s: %s", arxiv_id, exc)
                continue

        if fetched:
            store_cached_payloads(
                fetched,
                source=self.source,
                paper_by_arxiv_id=paper_by_arxiv_id,
                ttl_hours=self.ttl_hours,
            )
        return {**cached, **fetched}


def fetch_huggingface_batch(
    arxiv_ids: list[str],
    session: requests.Session | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch Hugging Face Papers data for a batch of arXiv IDs (cache-aware).

    Returns {arxiv_id: {hf_upvotes, hf_comments_count, github_repo_url, project_page_url}};
    a cached 404 miss maps to an empty dict.
    """
    provider = HuggingFaceProvider()
    return provider.fetch_batch(arxiv_ids, session=session)
