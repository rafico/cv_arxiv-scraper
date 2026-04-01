"""arXiv API ingest backend."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from typing import Any

import arxiv
import requests

from app.constants import ARXIV_API_BATCH_SIZE as _ARXIV_API_BATCH_SIZE
from app.constants import ARXIV_API_DELAY as _ARXIV_API_DELAY
from app.services.ingest.base import PaperCandidate, clean_abstract, extract_arxiv_id
from app.services.text import clean_whitespace

LOGGER = logging.getLogger(__name__)


class ArxivApiBackend:
    def __init__(self, *, page_size: int = _ARXIV_API_BATCH_SIZE, delay_seconds: float = _ARXIV_API_DELAY):
        self.page_size = page_size
        self.delay_seconds = delay_seconds

    @property
    def name(self) -> str:
        return "arxiv_api"

    @staticmethod
    def result_to_candidate(result: arxiv.Result) -> PaperCandidate:
        authors_list = [author.name for author in result.authors]
        publication_dt = result.published.date() if result.published else None
        publication_date = publication_dt.isoformat() if publication_dt else "Date Unknown"

        return PaperCandidate(
            arxiv_id=extract_arxiv_id(result.entry_id),
            link=result.entry_id,
            title=clean_whitespace(result.title),
            author=", ".join(authors_list),
            authors_list=authors_list,
            abstract=clean_abstract(result.summary),
            published=result.published.isoformat() if result.published else None,
            publication_dt=publication_dt,
            publication_date=publication_date,
            categories=list(result.categories or []),
            comment=result.comment or "",
            doi=result.doi or "",
        )

    def fetch(
        self,
        *,
        categories: Sequence[str],
        start_dt: date,
        end_dt: date,
        max_results: int = 1000,
        session: requests.Session | None = None,
        **kwargs: Any,
    ) -> list[PaperCandidate]:
        del session, kwargs

        if not categories:
            return []

        client = arxiv.Client(page_size=self.page_size, delay_seconds=self.delay_seconds, num_retries=3)
        cat_query = " OR ".join(f"cat:{category}" for category in categories)
        from_ts = start_dt.strftime("%Y%m%d0000")
        to_ts = end_dt.strftime("%Y%m%d2359")
        query_str = f"({cat_query}) AND submittedDate:[{from_ts} TO {to_ts}]"

        search = arxiv.Search(
            query=query_str,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        results: list[PaperCandidate] = []
        try:
            for result in client.results(search):
                results.append(self.result_to_candidate(result))
        except Exception as exc:
            LOGGER.error("arxiv.py client failed: %s", exc)

        return results
