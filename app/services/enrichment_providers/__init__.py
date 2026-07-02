"""Cache-aware enrichment provider exports."""

from app.services.enrichment_providers.base import EnrichmentProvider
from app.services.enrichment_providers.github import GitHubProvider, extract_github_repo
from app.services.enrichment_providers.huggingface import (
    HuggingFaceProvider,
    fetch_huggingface_batch,
    huggingface_resource_links,
    parse_hf_paper,
)
from app.services.enrichment_providers.openalex_provider import OpenAlexProvider, parse_openalex_work
from app.services.enrichment_providers.semantic_scholar import SemanticScholarProvider

__all__ = [
    "EnrichmentProvider",
    "GitHubProvider",
    "HuggingFaceProvider",
    "OpenAlexProvider",
    "SemanticScholarProvider",
    "extract_github_repo",
    "fetch_huggingface_batch",
    "huggingface_resource_links",
    "parse_hf_paper",
    "parse_openalex_work",
]
