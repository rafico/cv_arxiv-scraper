from flask import Blueprint, abort, redirect, url_for
from jinja2 import TemplateNotFound

from app.ui import render_ui

help_bp = Blueprint("help", __name__)


@help_bp.route("/help")
def view_help():
    return redirect(url_for("help.view_page", page="start"))


@help_bp.route("/help/<page>")
def view_page(page: str):
    try:
        if page not in ["start", "ui", "search", "organization", "features", "export", "cli", "settings", "faq"]:
            abort(404)
        return render_ui(f"help/{page}.html", active_page=page)
    except TemplateNotFound:
        abort(404)
