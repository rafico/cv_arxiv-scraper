"""Backward-compatible scraper facade.

The implementation lives in the service layer (`app.services`).
"""

from app.services.scrape_engine import run_scrape
from app.services.scrape_engine import stream_or_start_scrape as run_scrape_stream

__all__ = ["run_scrape", "run_scrape_stream"]
