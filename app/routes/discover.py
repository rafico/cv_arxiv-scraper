"""Discover page — UI for historical arXiv search."""

from flask import Blueprint, render_template

from app.csrf import get_or_create_csrf_token

discover_bp = Blueprint("discover", __name__)


@discover_bp.route("/discover")
def index():
    return render_template(
        "discover.html",
        csrf_token=get_or_create_csrf_token(),
    )
