"""Saved search CRUD and execution endpoints."""

from flask import abort, jsonify, request

from app.csrf import validate_csrf_token
from app.models import SavedSearch, db
from app.routes.api import api_bp
from app.routes.api._validation import optional_str, require_str


def _serialize_saved_search(s: SavedSearch) -> dict:
    return {
        "id": s.id,
        "name": s.name,
        "filters": s.filters,
        "categories": s.categories or [],
        "include_keywords": s.include_keywords or [],
        "exclude_keywords": s.exclude_keywords or [],
        "author_filters": s.author_filters or [],
        "date_window_days": s.date_window_days,
        "min_citations": s.min_citations,
        "methods_mentions": s.methods_mentions or [],
        "is_active": s.is_active,
        "notify_on_match": s.notify_on_match,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "last_used_at": s.last_used_at.isoformat() if s.last_used_at else None,
    }


@api_bp.route("/saved-searches", methods=["GET"])
def list_saved_searches():
    searches = SavedSearch.query.order_by(SavedSearch.created_at.desc()).all()
    return jsonify([_serialize_saved_search(s) for s in searches])


@api_bp.route("/saved-searches", methods=["POST"])
def create_saved_search():
    from app.services.saved_search import validate_saved_search

    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    # require_str yields a clean 400 ("Missing 'name'" / "'name' must be a string")
    # via the api_bp HTTPException handler instead of an AttributeError on a
    # non-string name falling through to the generic safety-net 400 + log noise.
    name = require_str(payload, "name")

    errors = validate_saved_search(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    filters = payload.get("filters", {})
    if not isinstance(filters, dict):
        return jsonify({"error": "'filters' must be a dict"}), 400

    s = SavedSearch(
        name=name,
        filters=filters,
        categories=payload.get("categories", []),
        include_keywords=payload.get("include_keywords", []),
        exclude_keywords=payload.get("exclude_keywords", []),
        author_filters=payload.get("author_filters", []),
        date_window_days=payload.get("date_window_days"),
        min_citations=payload.get("min_citations"),
        methods_mentions=payload.get("methods_mentions", []),
        is_active=payload.get("is_active", True),
        notify_on_match=payload.get("notify_on_match", False),
    )
    db.session.add(s)
    db.session.commit()
    return jsonify(_serialize_saved_search(s)), 201


@api_bp.route("/saved-searches/<int:search_id>", methods=["GET"])
def get_saved_search(search_id: int):
    s = db.session.get(SavedSearch, search_id) or abort(404)
    return jsonify(_serialize_saved_search(s))


@api_bp.route("/saved-searches/<int:search_id>", methods=["PUT"])
def update_saved_search(search_id: int):
    from app.services.saved_search import validate_saved_search

    validate_csrf_token()
    s = db.session.get(SavedSearch, search_id) or abort(404)
    payload = request.get_json(silent=True) or {}

    errors = validate_saved_search(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    if "name" in payload:
        # optional_str tolerates an absent/empty name (left unchanged) but rejects a
        # non-string with a clean 400 instead of an AttributeError.
        name = optional_str(payload, "name")
        if name:
            s.name = name
    if "filters" in payload:
        if not isinstance(payload["filters"], dict):
            return jsonify({"error": "'filters' must be a dict"}), 400
        s.filters = payload["filters"]
    for field in ("categories", "include_keywords", "exclude_keywords", "author_filters", "methods_mentions"):
        if field in payload:
            setattr(s, field, payload[field])
    if "date_window_days" in payload:
        s.date_window_days = payload["date_window_days"]
    if "min_citations" in payload:
        s.min_citations = payload["min_citations"]
    if "is_active" in payload:
        s.is_active = bool(payload["is_active"])
    if "notify_on_match" in payload:
        s.notify_on_match = bool(payload["notify_on_match"])

    db.session.commit()
    return jsonify(_serialize_saved_search(s))


@api_bp.route("/saved-searches/<int:search_id>", methods=["DELETE"])
def delete_saved_search(search_id: int):
    validate_csrf_token()
    s = db.session.get(SavedSearch, search_id) or abort(404)
    db.session.delete(s)
    db.session.commit()
    return jsonify({"deleted": True})


@api_bp.route("/saved-searches/<int:search_id>/run", methods=["POST"])
def run_saved_search(search_id: int):
    from app.services.saved_search import execute_saved_search

    validate_csrf_token()
    s = db.session.get(SavedSearch, search_id) or abort(404)
    payload = request.get_json(silent=True) or {}
    try:
        # Clamp the lower bound too: SQLite treats a negative LIMIT as "unlimited",
        # so a negative value would defeat the cap and return the whole corpus.
        limit = max(1, min(int(payload.get("limit", 100)), 500))
    except (ValueError, TypeError):
        limit = 100

    papers = execute_saved_search(s, limit=limit)

    s.last_used_at = db.func.now()
    db.session.commit()

    return jsonify(
        {
            "search_id": s.id,
            "search_name": s.name,
            "count": len(papers),
            "results": [
                {
                    "id": p.id,
                    "arxiv_id": p.arxiv_id,
                    "title": p.title,
                    "authors": p.authors,
                    "abstract": (p.abstract_text or "")[:300],
                    "paper_score": float(p.paper_score or 0),
                    "publication_dt": p.publication_dt.isoformat() if p.publication_dt else None,
                    "citation_count": p.citation_count,
                }
                for p in papers
            ],
        }
    )
