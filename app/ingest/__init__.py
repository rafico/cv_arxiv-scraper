"""Semantic package for ingestion and scraping workflows."""

from app.services.arxiv_adapter import result_to_entry
from app.services.enrichment import (
    enrich_entries_with_api_metadata,
    extract_affiliation_text,
    fetch_recent_papers,
    parse_feed_entries,
    query_arxiv_api,
)
from app.services.http_client import create_session, request_with_backoff, resolve_user_agent
from app.services.ingest import (
    ArxivApiBackend,
    IngestBackend,
    IngestMode,
    IngestOrchestrator,
    OaiPmhBackend,
    PaperCandidate,
    RssFeedBackend,
    SyncCursor,
)
from app.services.ingest.base import clean_abstract, extract_arxiv_id, parse_publication_dt
from app.services.ingest.orchestrator import BACKEND_REGISTRY
from app.services.rate_limiter import TokenBucketRateLimiter, get_shared_rate_limiter, resolve_rate_limit_settings
from app.services.scrape_engine import execute_historical_scrape, execute_scrape, run_scrape, stream_or_start_scrape

__all__ = [
    "ArxivApiBackend",
    "BACKEND_REGISTRY",
    "IngestBackend",
    "IngestMode",
    "IngestOrchestrator",
    "OaiPmhBackend",
    "PaperCandidate",
    "RssFeedBackend",
    "SyncCursor",
    "TokenBucketRateLimiter",
    "clean_abstract",
    "create_session",
    "enrich_entries_with_api_metadata",
    "execute_historical_scrape",
    "execute_scrape",
    "extract_affiliation_text",
    "extract_arxiv_id",
    "fetch_recent_papers",
    "get_shared_rate_limiter",
    "parse_feed_entries",
    "parse_publication_dt",
    "query_arxiv_api",
    "request_with_backoff",
    "resolve_rate_limit_settings",
    "resolve_user_agent",
    "result_to_entry",
    "run_scrape",
    "stream_or_start_scrape",
]
