"""Cache-aware enrichment provider exports."""

from app.services.enrichment_providers.base import EnrichmentProvider
from app.services.enrichment_providers.github import GitHubProvider, extract_github_repo
from app.services.enrichment_providers.openalex_provider import OpenAlexProvider, parse_openalex_work
from app.services.enrichment_providers.semantic_scholar import SemanticScholarProvider

__all__ = [
    "EnrichmentProvider",
    "GitHubProvider",
    "OpenAlexProvider",
    "SemanticScholarProvider",
    "extract_github_repo",
    "parse_openalex_work",
]
