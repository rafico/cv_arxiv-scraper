"""Ingestion backends and shared paper candidate types."""

from app.services.ingest.arxiv_api_backend import ArxivApiBackend
from app.services.ingest.base import IngestBackend, IngestMode, PaperCandidate
from app.services.ingest.oai_pmh_backend import OaiPmhBackend
from app.services.ingest.orchestrator import IngestOrchestrator
from app.services.ingest.rss_backend import RssFeedBackend

__all__ = [
    "ArxivApiBackend",
    "IngestBackend",
    "IngestMode",
    "IngestOrchestrator",
    "OaiPmhBackend",
    "PaperCandidate",
    "RssFeedBackend",
]
