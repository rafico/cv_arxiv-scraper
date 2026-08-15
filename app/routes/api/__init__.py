"""JSON API under /api, assembled from feature modules.

Each submodule attaches its routes to the shared `api_bp` blueprint when
imported below, so URLs and endpoint names match the former single-module
layout. `SCRAPE_JOB_MANAGER` is re-exported because tests patch it via this
module path.
"""

from flask import Blueprint, current_app, jsonify
from werkzeug.exceptions import HTTPException

from app.services import SCRAPE_JOB_MANAGER

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.record
def _register_sibling_blueprints(state) -> None:
    """Register top-level blueprints that must live *outside* the ``/api`` prefix.

    Uses ``api_bp``'s own registration event (fired by ``app.register_blueprint``
    in the app factory) so ``/healthz`` can be wired without editing
    ``app/__init__.py``. ``record`` (not ``record_once``) runs on every app
    registration, so each ``create_app()`` — including the many the test suite
    builds — gets its own copy. The membership guard keeps a re-registered
    ``api_bp`` from raising a duplicate-blueprint error.
    """
    from app.routes.health import health_bp

    if health_bp.name not in state.app.blueprints:
        state.app.register_blueprint(health_bp)


@api_bp.errorhandler(HTTPException)
def _handle_http_exception(exc: HTTPException):
    """Render HTTP errors (e.g. BadRequest from the validation helpers) as JSON."""
    return jsonify({"error": exc.description}), exc.code or 500


@api_bp.errorhandler(TypeError)
@api_bp.errorhandler(ValueError)
@api_bp.errorhandler(KeyError)
@api_bp.errorhandler(AttributeError)
@api_bp.errorhandler(OverflowError)
def _handle_bad_input(exc: Exception):
    """Safety net behind per-route validation: turn input-type errors into a clean 400.

    Per-route guards remain the primary defense; this catches anything they miss
    (e.g. a wrong-typed JSON body, or an oversized integer id that overflows
    SQLite's signed-64-bit range in a ``get``/``IN`` lookup) so it returns 400
    instead of an opaque 500. The exception is logged so genuine bugs stay visible.
    """
    current_app.logger.exception("Unhandled %s in API route", type(exc).__name__)
    return jsonify({"error": "Invalid request"}), 400


# Route modules attach their handlers to api_bp on import.
from app.routes.api import (  # noqa: E402
    annotations,
    backup,
    chat,
    collections,
    export,
    feed_sources,
    onboarding,
    one_tap,
    papers,
    profiles,
    saved_searches,
    scrape,
    search,
)

__all__ = [
    "SCRAPE_JOB_MANAGER",
    "annotations",
    "api_bp",
    "backup",
    "chat",
    "collections",
    "export",
    "feed_sources",
    "onboarding",
    "one_tap",
    "papers",
    "profiles",
    "saved_searches",
    "scrape",
    "search",
]
