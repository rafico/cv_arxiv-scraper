"""Collections CRUD and membership endpoints."""

from flask import abort, jsonify, request
from sqlalchemy.exc import IntegrityError

from app.csrf import validate_csrf_token
from app.models import Collection, Paper, PaperCollection, db
from app.routes.api import api_bp
from app.routes.api._validation import optional_str, require_list, require_str


@api_bp.route("/collections", methods=["GET"])
def list_collections():
    paper_count_subquery = (
        db.session.query(
            PaperCollection.collection_id,
            db.func.count(PaperCollection.id).label("paper_count"),
        )
        .group_by(PaperCollection.collection_id)
        .subquery()
    )
    results = (
        db.session.query(Collection, db.func.coalesce(paper_count_subquery.c.paper_count, 0))
        .outerjoin(paper_count_subquery, Collection.id == paper_count_subquery.c.collection_id)
        .order_by(Collection.name)
        .all()
    )
    return jsonify(
        [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description or "",
                "color": c.color,
                "paper_count": count,
            }
            for c, count in results
        ]
    )


@api_bp.route("/collections", methods=["POST"])
def create_collection():
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    name = require_str(payload, "name")
    if Collection.query.filter_by(name=name).first():
        return jsonify({"error": "Collection already exists"}), 409
    c = Collection(
        name=name,
        description=optional_str(payload, "description"),
        color=optional_str(payload, "color") or None,
    )
    db.session.add(c)
    db.session.commit()
    return jsonify({"id": c.id, "name": c.name}), 201


@api_bp.route("/collections/<int:collection_id>", methods=["PUT"])
def update_collection(collection_id: int):
    validate_csrf_token()
    c = db.session.get(Collection, collection_id) or abort(404)
    payload = request.get_json(silent=True) or {}
    name = optional_str(payload, "name")
    if name:
        if Collection.query.filter(Collection.name == name, Collection.id != c.id).first():
            return jsonify({"error": "Collection already exists"}), 409
        c.name = name
    if "description" in payload:
        c.description = optional_str(payload, "description")
    if "color" in payload:
        c.color = optional_str(payload, "color") or None
    try:
        db.session.commit()
    except IntegrityError:
        # Closes the (single-worker, narrow) race between the pre-check and commit.
        db.session.rollback()
        return jsonify({"error": "Collection already exists"}), 409
    return jsonify({"id": c.id, "name": c.name})


@api_bp.route("/collections/<int:collection_id>", methods=["DELETE"])
def delete_collection(collection_id: int):
    validate_csrf_token()
    c = db.session.get(Collection, collection_id) or abort(404)
    db.session.delete(c)
    db.session.commit()
    return jsonify({"deleted": True})


@api_bp.route("/collections/<int:collection_id>/papers", methods=["POST"])
def add_paper_to_collection(collection_id: int):
    validate_csrf_token()
    c = db.session.get(Collection, collection_id) or abort(404)
    payload = request.get_json(silent=True) or {}
    # ``bool`` is a subclass of ``int``: a JSON ``true``/``false`` would otherwise
    # resolve to ``Paper`` id 1/0 and silently add the wrong paper (mirrors the
    # bulk_feedback guard).
    if isinstance(payload.get("paper_id"), int) and not isinstance(payload.get("paper_id"), bool):
        paper_ids = [payload["paper_id"]]
    else:
        paper_ids = require_list(payload, "paper_ids")
    added = _stage_new_memberships(paper_ids, c.id)
    try:
        db.session.commit()
    except IntegrityError:
        # A concurrent request already inserted one of these rows (the narrow
        # pre-check/commit race under multiple threads); roll back and re-run
        # the toggle so the now-present rows are treated as no-ops and the
        # end-state (papers in collection) is reached idempotently, mirroring
        # create_collection/update_collection instead of an opaque 500.
        db.session.rollback()
        added = _stage_new_memberships(paper_ids, c.id)
        db.session.commit()
    return jsonify({"added": added, "collection_id": c.id})


def _stage_new_memberships(paper_ids: list, collection_id: int) -> int:
    """Stage PaperCollection rows for papers not already in the collection.

    Returns the count staged (not yet committed); skips non-int/bool ids,
    missing papers, and existing memberships."""
    added = 0
    for pid in paper_ids:
        if not isinstance(pid, int) or isinstance(pid, bool) or not db.session.get(Paper, pid):
            continue
        if PaperCollection.query.filter_by(paper_id=pid, collection_id=collection_id).first():
            continue
        db.session.add(PaperCollection(paper_id=pid, collection_id=collection_id))
        added += 1
    return added


@api_bp.route("/collections/<int:collection_id>/papers/<int:paper_id>", methods=["DELETE"])
def remove_paper_from_collection(collection_id: int, paper_id: int):
    validate_csrf_token()
    pc = PaperCollection.query.filter_by(paper_id=paper_id, collection_id=collection_id).first()
    if not pc:
        abort(404)
    db.session.delete(pc)
    db.session.commit()
    return jsonify({"removed": True})
