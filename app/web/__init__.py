"""Semantic package for web-facing integrations and schedulers."""

from app.services.bibtex import paper_to_bibtex, papers_to_bibtex
from app.services.cron import get_cron_status, install_cron_job, remove_cron_job
from app.services.email_digest import (
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_TOKEN_PATH,
    build_digest_preview,
    check_gmail_auth_status,
    finish_oauth_flow,
    send_digest,
    start_oauth_flow,
    validate_credentials_redirect_uris,
)
from app.services.jobs import SCRAPE_JOB_MANAGER, ScrapeJobManager
from app.services.llm_client import LLMClient, has_api_key, resolve_api_key, write_api_key
from app.services.mendeley import MendeleyClient
from app.services.scheduler import SCRAPE_SCHEDULER, ScrapeScheduler
from app.services.zotero import ZoteroClient

__all__ = [
    "DEFAULT_CREDENTIALS_PATH",
    "DEFAULT_TOKEN_PATH",
    "LLMClient",
    "MendeleyClient",
    "SCRAPE_JOB_MANAGER",
    "SCRAPE_SCHEDULER",
    "ScrapeJobManager",
    "ScrapeScheduler",
    "ZoteroClient",
    "build_digest_preview",
    "check_gmail_auth_status",
    "finish_oauth_flow",
    "get_cron_status",
    "has_api_key",
    "install_cron_job",
    "paper_to_bibtex",
    "papers_to_bibtex",
    "remove_cron_job",
    "resolve_api_key",
    "send_digest",
    "start_oauth_flow",
    "validate_credentials_redirect_uris",
    "write_api_key",
]
