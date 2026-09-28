"""Semantic Scholar enrichment provider."""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import TYPE_CHECKING, Any

from app.services.enrichment_providers.base import (
    DEFAULT_CACHE_TTL_HOURS,
    EnrichmentProvider,
    get_cached_payloads,
    store_cached_payloads,
)

if TYPE_CHECKING:
    from app.services.ingest.base import PaperCandidate

LOGGER = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
SEMANTIC_SCHOLAR_BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

# The Semantic Scholar batch endpoint caps each request at 500 ids; larger
# payloads error (400/413). Chunk to stay under the cap.
SEMANTIC_SCHOLAR_BATCH_LIMIT = 500


class SemanticScholarProvider(EnrichmentProvider):
    source = "semantic_scholar"

    def __init__(
        self, *, ttl_hours: int = DEFAULT_CACHE_TTL_HOURS, request_fn=None, api_key: str | None = None
    ) -> None:
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

        # references.paperId feeds the local citation graph: S2 parses reference
        # lists for fresh arXiv preprints within days, where OpenAlex lags months.
        params = {"fields": "citationCount,influentialCitationCount,paperId,references.paperId"}
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
                        "references": [
                            ref["paperId"] for ref in item.get("references") or [] if ref and ref.get("paperId")
                        ],
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


_S2_OPERATORS = {"AND": "+", "OR": "|", "ANDNOT": "-"}


def arxiv_query_to_s2(query: str) -> str:
    """Rewrite an arXiv API search query in Semantic Scholar bulk-search syntax.

    ponytail: term-level rewrite. Field prefixes are dropped (S2 matches title and
    abstract together, so an ``au:`` name becomes a plain word) and ``cat:`` terms are
    dropped along with the operators and groups they leave dangling (the caller narrows
    to Computer Science instead). Deeply nested queries translate loosely; upgrade to a
    real parser if people lean on this fallback.
    """
    out: list[str] = []
    for term in re.findall(r'(?:\w+:)?"[^"]*"|[()]|[^\s()]+', query):
        term = re.sub(r"^(?:all|ti|abs|au):", "", term)
        if not term or term.startswith("cat:"):
            continue
        op = _S2_OPERATORS.get(term)
        if op and (not out or out[-1] in ("(", *_S2_OPERATORS.values())):
            continue  # nothing on its left: the term before it was a dropped cat:
        if term == ")":
            if out and out[-1] in _S2_OPERATORS.values():
                out.pop()
            if out and out[-1] == "(":
                out.pop()
                continue
        out.append(op or term)
    while out and out[-1] in ("(", *_S2_OPERATORS.values()):
        out.pop()
    return " ".join(out).replace("- ", "-")


def search_arxiv_papers(query: str, start_dt: date, end_dt: date, max_results: int) -> list[PaperCandidate]:
    """arXiv papers matching an arXiv-syntax ``query`` via S2 bulk search, newest first.

    The fallback for ``cv-arxiv-sync --query`` when arXiv refuses the search. Hits S2 does
    not link to an arXiv id are skipped; S2 has no arXiv categories, so results are narrowed
    to Computer Science and come back without categories.
    """
    from app.services.http_client import request_with_backoff
    from app.services.ingest.base import PaperCandidate, extract_arxiv_id, parse_publication_dt
    from app.services.secret_files import resolve_data_source_key

    api_key = resolve_data_source_key("semantic_scholar")
    params = {
        "query": arxiv_query_to_s2(query),
        "fields": "externalIds,title,abstract,authors,publicationDate",
        "publicationDateOrYear": f"{start_dt}:{end_dt}",
        "fieldsOfStudy": "Computer Science",
        "sort": "publicationDate:desc",
    }
    candidates: list[PaperCandidate] = []
    while len(candidates) < max_results:
        data = request_with_backoff(
            "GET",
            SEMANTIC_SCHOLAR_BULK_SEARCH_URL,
            params=params,
            headers={"x-api-key": api_key} if api_key else None,
            rate_limit_profile="bulk",
        ).json()
        for item in data.get("data") or []:
            arxiv_id = extract_arxiv_id(f"https://arxiv.org/abs/{(item.get('externalIds') or {}).get('ArXiv') or ''}")
            if not arxiv_id:
                continue
            authors = [a["name"] for a in item.get("authors") or [] if a.get("name")]
            publication_dt, publication_date = parse_publication_dt(item.get("publicationDate"))
            candidates.append(
                PaperCandidate(
                    arxiv_id=arxiv_id,
                    link=f"https://arxiv.org/abs/{arxiv_id}",
                    title=item.get("title") or "",
                    author=", ".join(authors),
                    authors_list=authors,
                    abstract=item.get("abstract") or "",
                    publication_dt=publication_dt,
                    publication_date=publication_date,
                )
            )
        if not data.get("token"):
            break
        params["token"] = data["token"]
    return candidates[:max_results]
