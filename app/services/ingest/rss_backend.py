"""RSS ingest backend."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import feedparser
import requests

from app.services.http_client import request_with_backoff
from app.services.ingest.base import (
    PaperCandidate,
    clean_abstract,
    extract_arxiv_id,
    extract_author_names,
    parse_publication_dt,
)
from app.services.text import clean_whitespace

LOGGER = logging.getLogger(__name__)


class RssFeedBackend:
    def __init__(self, feed_urls: Iterable[str]):
        self.feed_urls = [url for url in feed_urls if url]

    @property
    def name(self) -> str:
        return "rss_feed"

    def fetch(self, *, session: requests.Session | None = None, **kwargs) -> list[PaperCandidate]:
        entries: list[PaperCandidate] = []

        for feed_url in self.feed_urls:
            response = request_with_backoff("GET", feed_url, timeout=30, session=session)
            feed = feedparser.parse(response.content)

            for entry in feed.entries:
                publication_dt, publication_date = parse_publication_dt(getattr(entry, "published", None))
                link = getattr(entry, "link", "")
                entries.append(
                    PaperCandidate(
                        arxiv_id=extract_arxiv_id(link),
                        link=link,
                        title=clean_whitespace(getattr(entry, "title", "")),
                        author=getattr(entry, "author", ""),
                        authors_list=extract_author_names(entry),
                        abstract=clean_abstract(getattr(entry, "summary", "")),
                        published=getattr(entry, "published", None),
                        publication_dt=publication_dt,
                        publication_date=publication_date,
                    )
                )

        LOGGER.info("Total entries across RSS feeds: %s", len(entries))
        return entries
