"""PDF reader annotations CRUD."""

from flask import abort, jsonify, request

from app.csrf import validate_csrf_token
from app.models import Paper, PaperAnnotation, db
from app.routes.api import api_bp

ANNOTATION_KINDS = {"highlight", "comment"}
MAX_NOTE_CHARS = 20_000
MAX_RECTS = 200


def _annotation_to_dict(a: PaperAnnotation) -> dict:
    return {
        "id": a.id,
        "paper_id": a.paper_id,
        "page": a.page,
        "kind": a.kind,
        "rects": a.rects,
        "color": a.color,
        "note": a.note,
    }


def _validated_rects(value: object) -> list:
    if not isinstance(value, list) or not value or len(value) > MAX_RECTS:
        raise ValueError("rects must be a non-empty list")
    for rect in value:
        if not isinstance(rect, dict) or set(rect) != {"x", "y", "w", "h"}:
            raise ValueError("each rect needs exactly x, y, w, h")
        for coord in rect.values():
            if not isinstance(coord, int | float) or isinstance(coord, bool) or not 0 <= coord <= 1:
                raise ValueError("rect coordinates must be numbers in [0, 1]")
    return value


@api_bp.route("/papers/<int:paper_id>/annotations", methods=["GET"])
def list_annotations(paper_id: int):
    db.session.get(Paper, paper_id) or abort(404)
    annotations = PaperAnnotation.query.filter_by(paper_id=paper_id).order_by(PaperAnnotation.id).all()
    return jsonify([_annotation_to_dict(a) for a in annotations])


@api_bp.route("/papers/<int:paper_id>/annotations", methods=["POST"])
def create_annotation(paper_id: int):
    validate_csrf_token()
    db.session.get(Paper, paper_id) or abort(404)
    payload = request.get_json(silent=True) or {}
    try:
        kind = payload.get("kind")
        if kind not in ANNOTATION_KINDS:
            raise ValueError(f"kind must be one of {sorted(ANNOTATION_KINDS)}")
        page = payload.get("page")
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise ValueError("page must be a positive integer")
        rects = _validated_rects(payload.get("rects"))
        note = payload.get("note") or ""
        if not isinstance(note, str) or len(note) > MAX_NOTE_CHARS:
            raise ValueError("note must be a string of reasonable length")
        color = payload.get("color")
        if color is not None and (not isinstance(color, str) or len(color) > 16):
            raise ValueError("invalid color")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    annotation = PaperAnnotation(paper_id=paper_id, page=page, kind=kind, rects=rects, color=color, note=note)
    db.session.add(annotation)
    db.session.commit()
    return jsonify(_annotation_to_dict(annotation)), 201


@api_bp.route("/annotations/<int:annotation_id>", methods=["PATCH"])
def update_annotation(annotation_id: int):
    validate_csrf_token()
    annotation = db.session.get(PaperAnnotation, annotation_id) or abort(404)
    payload = request.get_json(silent=True) or {}
    note = payload.get("note")
    if not isinstance(note, str) or len(note) > MAX_NOTE_CHARS:
        return jsonify({"error": "note must be a string of reasonable length"}), 400
    annotation.note = note
    db.session.commit()
    return jsonify(_annotation_to_dict(annotation))


@api_bp.route("/annotations/<int:annotation_id>", methods=["DELETE"])
def delete_annotation(annotation_id: int):
    validate_csrf_token()
    annotation = db.session.get(PaperAnnotation, annotation_id) or abort(404)
    db.session.delete(annotation)
    db.session.commit()
    return jsonify({"deleted": True})
