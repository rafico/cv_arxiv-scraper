"""JSON API under /api, assembled from feature modules.

Each submodule attaches its routes to the shared `api_bp` blueprint when
imported below, so URLs and endpoint names match the former single-module
layout. `SCRAPE_JOB_MANAGER` is re-exported because tests patch it via this
module path.
"""

from flask import Blueprint

from app.services import SCRAPE_JOB_MANAGER

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Route modules attach their handlers to api_bp on import.
from app.routes.api import (  # noqa: E402
    collections,
    export,
    feed_sources,
    papers,
    saved_searches,
    scrape,
    search,
)

__all__ = [
    "SCRAPE_JOB_MANAGER",
    "api_bp",
    "collections",
    "export",
    "feed_sources",
    "papers",
    "saved_searches",
    "scrape",
    "search",
]
