"""Adapter for arXiv-style result objects to match the project's ingest candidate structure."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.services.ingest.base import PaperCandidate, clean_abstract, extract_arxiv_id
from app.services.text import clean_whitespace


def _author_name(author: Any) -> str:
    if isinstance(author, str):
        return clean_whitespace(author)
    return clean_whitespace(getattr(author, "name", ""))


def result_to_candidate(result: Any) -> PaperCandidate:
    """Translate an arXiv-style result object into the typed ingest candidate structure."""
    authors_list = [name for name in (_author_name(author) for author in getattr(result, "authors", [])) if name]
    published = getattr(result, "published", None)
    publication_dt = published.date() if isinstance(published, datetime) else None
    publication_date = publication_dt.isoformat() if publication_dt else "Date Unknown"

    return PaperCandidate(
        arxiv_id=extract_arxiv_id(getattr(result, "entry_id", "")),
        link=getattr(result, "entry_id", ""),
        title=clean_whitespace(getattr(result, "title", "")),
        author=", ".join(authors_list),
        authors_list=authors_list,
        abstract=clean_abstract(getattr(result, "summary", "")),
        published=published.isoformat() if isinstance(published, datetime) else None,
        publication_dt=publication_dt,
        publication_date=publication_date,
        categories=list(getattr(result, "categories", []) or []),
        comment=getattr(result, "comment", "") or "",
        doi=getattr(result, "doi", "") or "",
    )


def result_to_entry(result: Any) -> dict:
    """Translate an arXiv-style result object into the dictionary format expected by the scraper."""
    return result_to_candidate(result).to_entry_dict()
