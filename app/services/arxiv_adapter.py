"""Adapter for arxiv.py library to match the project's entry dictionary structure."""

import arxiv

from app.services.enrichment import clean_abstract, extract_arxiv_id
from app.services.text import clean_whitespace


def result_to_entry(result: arxiv.Result) -> dict:
    """Translate an arxiv.Result object into the dictionary format expected by the scraper."""
    authors_list = [author.name for author in result.authors]
    pub_dt = result.published.date() if result.published else None
    pub_date_str = pub_dt.isoformat() if pub_dt else "Date Unknown"

    return {
        "arxiv_id": extract_arxiv_id(result.entry_id),
        "link": result.entry_id,
        "title": clean_whitespace(result.title),
        "author": ", ".join(authors_list),
        "authors_list": authors_list,
        "abstract": clean_abstract(result.summary),
        "published": result.published.isoformat() if result.published else None,
        "publication_dt": pub_dt,
        "publication_date": pub_date_str,
        "categories": result.categories,
        "comment": result.comment or "",
        "doi": result.doi or "",
    }
