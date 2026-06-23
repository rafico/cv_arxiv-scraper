"""Feed source management endpoints."""

from flask import abort, jsonify, request
from sqlalchemy.exc import IntegrityError

from app.csrf import validate_csrf_token
from app.models import db
from app.routes.api import api_bp
from app.routes.api._validation import optional_str, require_str


@api_bp.route("/feed-sources", methods=["GET"])
def list_feed_sources():
    from app.models import FeedSource

    sources = FeedSource.query.order_by(FeedSource.created_at).all()
    return jsonify(
        [
            {
                "id": s.id,
                "name": s.name,
                "url": s.url,
                "feed_type": s.feed_type,
                "enabled": s.enabled,
                "last_fetched_at": s.last_fetched_at.isoformat() if s.last_fetched_at else None,
            }
            for s in sources
        ]
    )


@api_bp.route("/feed-sources", methods=["POST"])
def create_feed_source():
    from app.models import FeedSource

    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    name = require_str(payload, "name")
    url = require_str(payload, "url")
    if FeedSource.query.filter_by(url=url).first():
        return jsonify({"error": "Feed source with this URL already exists"}), 409
    feed_type = optional_str(payload, "feed_type") or "arxiv_rss"
    s = FeedSource(name=name, url=url, feed_type=feed_type)
    db.session.add(s)
    try:
        db.session.commit()
    except IntegrityError:
        # Closes the (single-worker, narrow) race between the pre-check and commit.
        db.session.rollback()
        return jsonify({"error": "Feed source with this URL already exists"}), 409
    return jsonify({"id": s.id, "name": s.name}), 201


@api_bp.route("/feed-sources/<int:source_id>", methods=["DELETE"])
def delete_feed_source(source_id: int):
    from app.models import FeedSource

    validate_csrf_token()
    s = db.session.get(FeedSource, source_id) or abort(404)
    db.session.delete(s)
    db.session.commit()
    return jsonify({"deleted": True})


@api_bp.route("/feed-sources/<int:source_id>/toggle", methods=["POST"])
def toggle_feed_source(source_id: int):
    from app.models import FeedSource

    validate_csrf_token()
    s = db.session.get(FeedSource, source_id) or abort(404)
    s.enabled = not s.enabled
    db.session.commit()
    return jsonify({"id": s.id, "enabled": s.enabled})
