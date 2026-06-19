"""Discover page — UI for historical arXiv search."""

from flask import Blueprint

from app.csrf import get_or_create_csrf_token
from app.ui import render_ui

discover_bp = Blueprint("discover", __name__)


@discover_bp.route("/discover")
def index():
    return render_ui(
        "discover.html",
        csrf_token=get_or_create_csrf_token(),
    )
