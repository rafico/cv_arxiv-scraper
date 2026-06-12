"""Semantic package for external enrichment providers."""

from app.services.citations import fetch_citations_batch
from app.services.enrichment_providers import (
    EnrichmentProvider,
    GitHubProvider,
    OpenAlexProvider,
    SemanticScholarProvider,
    extract_github_repo,
    parse_openalex_work,
)
from app.services.openalex import fetch_openalex_batch

__all__ = [
    "EnrichmentProvider",
    "GitHubProvider",
    "OpenAlexProvider",
    "SemanticScholarProvider",
    "extract_github_repo",
    "fetch_citations_batch",
    "fetch_openalex_batch",
    "parse_openalex_work",
]
