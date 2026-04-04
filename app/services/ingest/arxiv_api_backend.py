"""arXiv API ingest backend."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from types import MethodType
from typing import Any

import arxiv
import feedparser
import requests

from app.constants import ARXIV_API_BATCH_SIZE as _ARXIV_API_BATCH_SIZE
from app.constants import ARXIV_API_DELAY as _ARXIV_API_DELAY
from app.services.ingest.base import PaperCandidate, clean_abstract, extract_arxiv_id
from app.services.text import clean_whitespace

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[int, PaperCandidate], None]


def _apply_user_agent_override(client: arxiv.Client, user_agent: str | None) -> None:
    """Override arxiv.py's hard-coded User-Agent for polite API access."""
    if not user_agent:
        return

    def _try_parse_feed(self, url: str, first_page: bool, try_index: int):
        if self._last_request_dt is not None:
            required = timedelta(seconds=self.delay_seconds)
            since_last_request = datetime.now() - self._last_request_dt
            if since_last_request < required:
                time.sleep((required - since_last_request).total_seconds())

        resp = self._session.get(url, headers={"user-agent": user_agent})
        self._last_request_dt = datetime.now()
        if resp.status_code != requests.codes.OK:
            raise arxiv.HTTPError(url, try_index, resp.status_code)

        feed = feedparser.parse(resp.content)
        if len(feed.entries) == 0 and not first_page:
            raise arxiv.UnexpectedEmptyPageError(url, try_index, feed)

        if feed.bozo:
            LOGGER.warning(
                "Bozo feed; consider handling: %s",
                feed.bozo_exception if "bozo_exception" in feed else None,
            )

        return feed

    client._Client__try_parse_feed = MethodType(_try_parse_feed, client)


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
        offset: int = 0,
        resume_after_arxiv_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
        user_agent: str | None = None,
        **kwargs: Any,
    ) -> list[PaperCandidate]:
        del session, kwargs

        if not categories:
            return []

        client = arxiv.Client(page_size=self.page_size, delay_seconds=self.delay_seconds, num_retries=3)
        _apply_user_agent_override(client, user_agent)
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
        start_offset = max(0, int(offset))
        resume_page = ((start_offset // self.page_size) + 1) if resume_after_arxiv_id else None
        resume_consumed = resume_after_arxiv_id is None
        try:
            for index, result in enumerate(client.results(search, offset=start_offset)):
                candidate = self.result_to_candidate(result)
                current_page = ((start_offset + index) // self.page_size) + 1

                if not resume_consumed and resume_page is not None:
                    if current_page > resume_page:
                        resume_consumed = True
                    elif candidate.arxiv_id == resume_after_arxiv_id:
                        resume_consumed = True
                        continue
                    else:
                        continue

                results.append(candidate)
                if progress_callback is not None:
                    progress_callback(current_page, candidate)
        except Exception as exc:
            LOGGER.error("arxiv.py client failed: %s", exc)

        return results
