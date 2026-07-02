"""Ranking-onboarding endpoints: cold-start bootstrap + active-learning loop."""

from flask import current_app, jsonify, request

from app.csrf import validate_csrf_token
from app.routes.api import api_bp
from app.routes.api._validation import parse_int_query_arg
from app.services.onboarding import bootstrap_from_arxiv_ids, select_uncertain_papers

# Cap accepted ids so a near-MAX_CONTENT_LENGTH body cannot drive a large
# synchronous per-id DB/embedding loop that pins one of the few worker threads
# (mirrors MAX_BULK_IDS in export.py).
MAX_ONBOARDING_IDS = 500


def _parse_arxiv_ids(payload: dict) -> list[str]:
    """Accept ``arxiv_ids`` as a list or a newline/comma-separated string."""
    raw = payload.get("arxiv_ids")
    if isinstance(raw, str):
        return [token.strip() for token in raw.replace(",", "\n").splitlines() if token.strip()]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, str) and item.strip()]
    return []


@api_bp.route("/onboarding/bootstrap", methods=["POST"])
def onboarding_bootstrap():
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    arxiv_ids = _parse_arxiv_ids(payload)
    if not arxiv_ids:
        return jsonify({"error": "Missing 'arxiv_ids'"}), 400
    if len(arxiv_ids) > MAX_ONBOARDING_IDS:
        return jsonify({"error": f"Too many arXiv IDs (max {MAX_ONBOARDING_IDS})"}), 400

    summary = bootstrap_from_arxiv_ids(arxiv_ids, app=current_app._get_current_object())
    return jsonify(summary)


@api_bp.route("/onboarding/uncertain", methods=["GET"])
def onboarding_uncertain():
    try:
        limit = parse_int_query_arg("limit", default=2, minimum=1, maximum=10)
    except ValueError:
        limit = 2  # this endpoint's contract: silently fall back on bad input
    min_saves = 3

    papers = select_uncertain_papers(limit=limit, min_saves=min_saves)

    from app.services.onboarding import _saved_paper_count

    saved_total = _saved_paper_count()
    return jsonify({"papers": papers, "saved_total": saved_total, "min_saves": min_saves})
