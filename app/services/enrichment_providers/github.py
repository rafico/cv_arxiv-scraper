"""GitHub repository metadata enrichment provider."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

import requests

from app.services.enrichment_providers.base import (
    EnrichmentProvider,
    get_cached_payloads,
    store_cached_payloads,
)

LOGGER = logging.getLogger(__name__)

GITHUB_REPO_API_URL = "https://api.github.com/repos/{repo}"
# Repo metadata changes slowly; 14 days keeps unauthenticated quota usage low.
GITHUB_CACHE_TTL_HOURS = 336
# Unauthenticated GitHub API allows 60 requests/hour.
DEFAULT_MAX_FETCHES_PER_RUN = 25

_REPO_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def extract_github_repo(resource_links: list[dict] | None) -> str | None:
    """Return ``owner/repo`` from the first GitHub code link, if any."""
    for link in resource_links or []:
        parsed = urlparse(link.get("url", ""))
        if parsed.netloc.lower() not in ("github.com", "www.github.com"):
            continue
        segments = [segment for segment in parsed.path.split("/") if segment]
        if len(segments) < 2:
            continue
        owner, repo = segments[0], segments[1].removesuffix(".git")
        if _REPO_SEGMENT_RE.match(owner) and _REPO_SEGMENT_RE.match(repo):
            return f"{owner}/{repo}"
    return None


def _is_rate_limited(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return response is not None and response.status_code in (403, 429)


def _is_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return response is not None and response.status_code == 404


class GitHubProvider(EnrichmentProvider):
    source = "github"

    def __init__(
        self,
        *,
        ttl_hours: int = GITHUB_CACHE_TTL_HOURS,
        request_fn=None,
        max_fetches: int = DEFAULT_MAX_FETCHES_PER_RUN,
        token: str | None = None,
    ) -> None:
        self.ttl_hours = ttl_hours
        self._request_fn = request_fn
        self.max_fetches = max_fetches
        self.token = token

    def fetch_batch(  # type: ignore[override]  # provider-specific kwargs; base Protocol uses **kwargs
        self,
        arxiv_ids: list[str],
        repos_by_arxiv_id: dict[str, str] | None = None,
        session: requests.Session | None = None,
    ) -> dict[str, dict[str, Any]]:
        from app.services.http_client import request_with_backoff

        if not arxiv_ids:
            return {}

        repos_by_arxiv_id = repos_by_arxiv_id or {}
        request_fn = self._request_fn or request_with_backoff
        cached, missing_ids, paper_by_arxiv_id = get_cached_payloads(arxiv_ids, source=self.source)

        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        fetched: dict[str, dict[str, Any]] = {}
        fetches = 0
        for arxiv_id in missing_ids:
            repo = repos_by_arxiv_id.get(arxiv_id)
            if not repo:
                continue
            if fetches >= self.max_fetches:
                LOGGER.info("GitHub fetch cap (%d) reached; deferring remaining repos", self.max_fetches)
                break
            fetches += 1

            try:
                response = request_fn(
                    "GET",
                    GITHUB_REPO_API_URL.format(repo=repo),
                    headers=headers,
                    session=session,
                    timeout=15,
                    attempts=2,
                    rate_limit_profile="bulk",
                )
            except Exception as exc:
                if _is_rate_limited(exc):
                    LOGGER.warning("GitHub API rate limited; skipping remaining repos: %s", exc)
                    break
                if _is_not_found(exc):
                    # Cache the miss so a deleted repo is not re-queried every run.
                    fetched[arxiv_id] = {"github_repo": repo, "github_stars": None, "github_license": None}
                    continue
                LOGGER.warning("GitHub metadata fetch failed for %s: %s", repo, exc)
                continue

            data = response.json()
            license_info = data.get("license") or {}
            fetched[arxiv_id] = {
                "github_repo": data.get("full_name") or repo,
                "github_stars": data.get("stargazers_count"),
                "github_license": license_info.get("spdx_id") or license_info.get("name"),
                "archived": data.get("archived"),
                "pushed_at": data.get("pushed_at"),
            }

        if fetched:
            store_cached_payloads(
                fetched,
                source=self.source,
                paper_by_arxiv_id=paper_by_arxiv_id,
                ttl_hours=self.ttl_hours,
            )
        return {**cached, **fetched}
