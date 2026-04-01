"""Adapter for arxiv.py library to match the project's entry dictionary structure."""

import arxiv

from app.services.ingest.arxiv_api_backend import ArxivApiBackend


def result_to_entry(result: arxiv.Result) -> dict:
    """Translate an arxiv.Result object into the dictionary format expected by the scraper."""
    return ArxivApiBackend.result_to_candidate(result).to_entry_dict()
