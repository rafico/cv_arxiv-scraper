"""Citation graph page — force-directed view of local "cites" edges."""

from flask import Blueprint, render_template

graph_bp = Blueprint("graph", __name__)


@graph_bp.route("/graph")
def index():
    return render_template("graph.html")
