"""Core ingestion types and normalization helpers."""

from __future__ import annotations

import email.utils
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Protocol

import feedparser
import requests

from app.services.text import clean_whitespace

_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/(.+?)(?:v\d+)?$")


def extract_arxiv_id(link: str) -> str | None:
    match = _ARXIV_ID_RE.search(link)
    return match.group(1) if match else None


def parse_publication_dt(published: str | None) -> tuple[date | None, str]:
    if not published:
        return None, "Date Unknown"

    try:
        parsed = email.utils.parsedate_to_datetime(published)
        parsed_date = parsed.date()
        return parsed_date, parsed_date.isoformat()
    except Exception:
        return None, "Date Unknown"


def clean_abstract(summary: str | None) -> str:
    if not summary:
        return ""
    no_html = re.sub(r"<[^>]+>", " ", summary)
    return clean_whitespace(no_html)


def extract_author_names(entry: feedparser.FeedParserDict) -> list[str]:
    if hasattr(entry, "authors") and entry.authors:
        names = [author.get("name", "") for author in entry.authors if author.get("name")]
        if names:
            return names

    raw_authors = getattr(entry, "author", "")
    return [name for name in re.split(r",\s*|\s+and\s+", raw_authors) if name]


class IngestMode(str, Enum):
    DAILY_WATCH = "daily_watch"
    BACKFILL = "backfill"
    CATCH_UP = "catch_up"


class IngestBackend(Protocol):
    @property
    def name(self) -> str: ...

    def fetch(self, *, session: requests.Session | None = None, **kwargs: Any) -> list[PaperCandidate]: ...


@dataclass(slots=True)
class PaperCandidate:
    arxiv_id: str | None
    link: str
    title: str
    author: str = ""
    authors_list: list[str] = field(default_factory=list)
    abstract: str = ""
    published: str | None = None
    publication_dt: date | None = None
    publication_date: str = "Date Unknown"
    categories: list[str] = field(default_factory=list)
    comment: str = ""
    doi: str = ""
    api_affiliations: str = ""
    resource_links: list[dict[str, str]] = field(default_factory=list)

    def to_entry_dict(self) -> dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "link": self.link,
            "title": self.title,
            "author": self.author,
            "authors_list": list(self.authors_list),
            "abstract": self.abstract,
            "published": self.published,
            "publication_dt": self.publication_dt,
            "publication_date": self.publication_date,
            "categories": list(self.categories),
            "comment": self.comment,
            "doi": self.doi,
            "api_affiliations": self.api_affiliations,
            "resource_links": [dict(resource) for resource in self.resource_links],
        }

    def to_entry(self) -> dict[str, Any]:
        return self.to_entry_dict()

    @classmethod
    def from_entry_dict(cls, entry: Mapping[str, Any]) -> PaperCandidate:
        publication_dt = entry.get("publication_dt")
        publication_date = entry.get("publication_date")
        if not publication_date:
            publication_date = publication_dt.isoformat() if publication_dt else "Date Unknown"

        return cls(
            arxiv_id=entry.get("arxiv_id"),
            link=entry.get("link", ""),
            title=entry.get("title", ""),
            author=entry.get("author", ""),
            authors_list=list(entry.get("authors_list") or []),
            abstract=entry.get("abstract", ""),
            published=entry.get("published"),
            publication_dt=publication_dt,
            publication_date=publication_date,
            categories=list(entry.get("categories") or []),
            comment=entry.get("comment", ""),
            doi=entry.get("doi", ""),
            api_affiliations=entry.get("api_affiliations", ""),
            resource_links=[dict(resource) for resource in (entry.get("resource_links") or [])],
        )

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> PaperCandidate:
        return cls.from_entry_dict(entry)
