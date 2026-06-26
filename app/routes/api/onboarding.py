"""Ranking-onboarding endpoints: cold-start bootstrap + active-learning loop."""

from flask import current_app, jsonify, request

from app.csrf import validate_csrf_token
from app.routes.api import api_bp
from app.services.onboarding import bootstrap_from_arxiv_ids, select_uncertain_papers


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

    summary = bootstrap_from_arxiv_ids(arxiv_ids, app=current_app._get_current_object())
    return jsonify(summary)


@api_bp.route("/onboarding/uncertain", methods=["GET"])
def onboarding_uncertain():
    try:
        limit = int(request.args.get("limit", 2))
    except (TypeError, ValueError):
        limit = 2
    limit = max(1, min(limit, 10))
    min_saves = 3

    papers = select_uncertain_papers(limit=limit, min_saves=min_saves)

    from app.models import PaperFeedback, db

    saved_total = int(
        db.session.query(db.func.count(db.func.distinct(PaperFeedback.paper_id)))
        .filter(PaperFeedback.action == "save")
        .scalar()
        or 0
    )
    return jsonify({"papers": papers, "saved_total": saved_total, "min_saves": min_saves})
