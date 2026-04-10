"""arXiv API ingest backend."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

import defusedxml.ElementTree as ET
import requests

from app.constants import ARXIV_API_BATCH_SIZE as _ARXIV_API_BATCH_SIZE
from app.constants import ARXIV_API_DELAY as _ARXIV_API_DELAY
from app.services.http_client import request_with_backoff
from app.services.ingest.base import PaperCandidate, clean_abstract, extract_arxiv_id, parse_publication_dt
from app.services.text import clean_whitespace

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[int, PaperCandidate], None]

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ARXIV_API_TIMEOUT = 45
_ARXIV_API_ATTEMPTS = 4
_ARXIV_API_BASE_DELAY = 2.0
_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _build_query(categories: Sequence[str], start_dt: date, end_dt: date) -> str:
    cat_query = " OR ".join(f"cat:{category}" for category in categories)
    from_ts = start_dt.strftime("%Y%m%d0000")
    to_ts = end_dt.strftime("%Y%m%d2359")
    return f"({cat_query}) AND submittedDate:[{from_ts} TO {to_ts}]"


def _parse_atom_candidate(entry: ET.Element) -> PaperCandidate:
    id_el = entry.find("atom:id", _ATOM_NS)
    title_el = entry.find("atom:title", _ATOM_NS)
    summary_el = entry.find("atom:summary", _ATOM_NS)
    published_el = entry.find("atom:published", _ATOM_NS)
    comment_el = entry.find("arxiv:comment", _ATOM_NS)
    doi_el = entry.find("arxiv:doi", _ATOM_NS)

    link = clean_whitespace(id_el.text if id_el is not None and id_el.text else "")
    authors_list = [
        clean_whitespace(name_el.text)
        for author_el in entry.findall("atom:author", _ATOM_NS)
        for name_el in [author_el.find("atom:name", _ATOM_NS)]
        if name_el is not None and name_el.text
    ]
    categories = [
        term
        for term in (category_el.get("term", "").strip() for category_el in entry.findall("atom:category", _ATOM_NS))
        if term
    ]
    published = clean_whitespace(published_el.text if published_el is not None and published_el.text else "")
    publication_dt, publication_date = parse_publication_dt(published or None)

    return PaperCandidate(
        arxiv_id=extract_arxiv_id(link),
        link=link,
        title=clean_whitespace(title_el.text if title_el is not None else ""),
        author=", ".join(authors_list),
        authors_list=authors_list,
        abstract=clean_abstract(summary_el.text if summary_el is not None else ""),
        published=published or None,
        publication_dt=publication_dt,
        publication_date=publication_date,
        categories=categories,
        comment=clean_whitespace(comment_el.text if comment_el is not None else ""),
        doi=clean_whitespace(doi_el.text if doi_el is not None else ""),
    )


class ArxivApiBackend:
    def __init__(self, *, page_size: int = _ARXIV_API_BATCH_SIZE, delay_seconds: float = _ARXIV_API_DELAY):
        self.page_size = page_size
        self.delay_seconds = delay_seconds

    @property
    def name(self) -> str:
        return "arxiv_api"

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
        del kwargs

        if not categories or max_results <= 0:
            return []

        query_str = _build_query(categories, start_dt, end_dt)
        results: list[PaperCandidate] = []
        start = max(0, int(offset))
        resume_page = ((start // self.page_size) + 1) if resume_after_arxiv_id else None
        resume_consumed = resume_after_arxiv_id is None

        try:
            while len(results) < max_results:
                if start > max(0, int(offset)) and self.delay_seconds > 0:
                    time.sleep(self.delay_seconds)

                batch_limit = min(self.page_size, max_results - len(results))
                response = request_with_backoff(
                    "GET",
                    _ARXIV_API_URL,
                    params={
                        "search_query": query_str,
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                        "start": start,
                        "max_results": batch_limit,
                    },
                    timeout=_ARXIV_API_TIMEOUT,
                    attempts=_ARXIV_API_ATTEMPTS,
                    base_delay=_ARXIV_API_BASE_DELAY,
                    rate_limit_profile="bulk",
                    session=session,
                    user_agent=user_agent,
                )
                root = ET.fromstring(response.text)
                entries = root.findall("atom:entry", _ATOM_NS)
                if not entries:
                    break

                for batch_index, entry in enumerate(entries):
                    candidate = _parse_atom_candidate(entry)
                    current_page = ((start + batch_index) // self.page_size) + 1

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
                    if len(results) >= max_results:
                        break

                if len(entries) < batch_limit:
                    break
                start += batch_limit
        except Exception as exc:
            LOGGER.error("Direct arXiv API query failed: %s", exc)

        return results
