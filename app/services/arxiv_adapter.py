"""Adapter for arxiv.py library to match the project's ingest candidate structure."""

from __future__ import annotations

import arxiv

from app.services.ingest.base import PaperCandidate, clean_abstract, extract_arxiv_id
from app.services.text import clean_whitespace


def result_to_candidate(result: arxiv.Result) -> PaperCandidate:
    """Translate an arxiv.Result object into the typed ingest candidate structure."""
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


def result_to_entry(result: arxiv.Result) -> dict:
    """Translate an arxiv.Result object into the dictionary format expected by the scraper."""
    return result_to_candidate(result).to_entry_dict()
