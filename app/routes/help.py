from __future__ import annotations

from flask import Blueprint, render_template

help_bp = Blueprint("help", __name__)


@help_bp.route("/help")
def view_help():
    return render_template("help.html")
