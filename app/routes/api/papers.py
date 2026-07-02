"""Per-paper actions: status, notes, tags, feedback, explanations, follow/mute."""

import threading

from flask import abort, current_app, jsonify, request

from app.csrf import validate_csrf_token
from app.enums import ReadingStatus
from app.models import Paper, db
from app.routes._config import config_write_lock, persist_config
from app.routes.api import api_bp
from app.routes.api._validation import require_str
from app.services import apply_feedback_action
from app.services.preferences import (
    append_muted_term,
    append_whitelist_term,
    first_author_name,
)

# Serializes the non-atomic read-modify-write on a paper's user_tags JSON list.
# SQLite (WAL) ignores SELECT ... FOR UPDATE, so two concurrent gthread requests
# would each read the same snapshot and last-write-wins would drop a tag; a
# process-local lock is sufficient for this single-worker app (mirrors the
# _config write lock).
_TAG_WRITE_LOCK = threading.Lock()


@api_bp.route("/papers/<int:paper_id>/mendeley", methods=["POST"])
def single_paper_mendeley(paper_id: int):
    validate_csrf_token()
    paper = db.session.get(Paper, paper_id) or abort(404, description="Paper not found")

    from app.services.mendeley import MendeleyClient

    client = MendeleyClient()
    status = client.check_connection()
    if status["status"] != "connected":
        return jsonify({"error": f"Mendeley not connected: {status['message']}"}), 400

    result = client.add_document(paper)
    if not result["success"]:
        return jsonify({"error": result["message"]}), 502

    doc_id = result.get("document_id")
    if doc_id:
        paper.mendeley_doc_id = str(doc_id)
        db.session.commit()

    return jsonify(
        {
            "paper_id": paper.id,
            "message": result["message"],
            "document_id": doc_id,
        }
    )


VALID_READING_STATUSES = {status.value for status in ReadingStatus}


@api_bp.route("/papers/<int:paper_id>/reading-status", methods=["POST"])
def paper_reading_status(paper_id: int):
    validate_csrf_token()
    paper = db.session.get(Paper, paper_id) or abort(404, description="Paper not found")
    payload = request.get_json(silent=True) or {}
    status = payload.get("status")
    if status is not None and not isinstance(status, str):
        # A non-hashable (dict/list) would raise TypeError on the set lookup below.
        return jsonify({"error": "'status' must be a string"}), 400
    if status is not None and status not in VALID_READING_STATUSES:
        return jsonify({"error": f"Invalid status. Must be one of: {', '.join(sorted(VALID_READING_STATUSES))}"}), 400
    paper.reading_status = status
    db.session.commit()
    return jsonify({"paper_id": paper.id, "reading_status": paper.reading_status})


@api_bp.route("/papers/<int:paper_id>/notes", methods=["PUT"])
def paper_notes(paper_id: int):
    validate_csrf_token()
    paper = db.session.get(Paper, paper_id) or abort(404, description="Paper not found")
    payload = request.get_json(silent=True) or {}
    notes = payload.get("notes", "")
    if not isinstance(notes, str):
        return jsonify({"error": "'notes' must be a string"}), 400
    paper.user_notes = notes
    db.session.commit()
    return jsonify({"paper_id": paper.id, "user_notes": paper.user_notes})


@api_bp.route("/papers/<int:paper_id>/tags", methods=["POST"])
def paper_add_tag(paper_id: int):
    validate_csrf_token()
    paper = db.session.get(Paper, paper_id) or abort(404, description="Paper not found")
    payload = request.get_json(silent=True) or {}
    tag = require_str(payload, "tag")
    with _TAG_WRITE_LOCK:
        # Re-read inside the lock so the read-modify-write sees any tag a
        # concurrent request just committed, instead of clobbering it.
        db.session.refresh(paper)
        current = list(paper.user_tags or [])
        if tag not in current:
            current.append(tag)
            paper.user_tags = current
            db.session.commit()
    return jsonify({"paper_id": paper.id, "user_tags": paper.user_tags})


@api_bp.route("/papers/<int:paper_id>/tags", methods=["DELETE"])
def paper_remove_tag(paper_id: int):
    validate_csrf_token()
    paper = db.session.get(Paper, paper_id) or abort(404, description="Paper not found")
    payload = request.get_json(silent=True) or {}
    tag = require_str(payload, "tag")
    with _TAG_WRITE_LOCK:
        # Re-read inside the lock so the read-modify-write sees any tag a
        # concurrent request just committed, instead of clobbering it.
        db.session.refresh(paper)
        current = list(paper.user_tags or [])
        if tag in current:
            current.remove(tag)
            paper.user_tags = current
            db.session.commit()
    return jsonify({"paper_id": paper.id, "user_tags": paper.user_tags})


@api_bp.route("/papers/<int:paper_id>/feedback", methods=["POST"])
def paper_feedback(paper_id: int):
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    action = payload.get("action")
    if not isinstance(action, str):
        return jsonify({"error": "Missing 'action'"}), 400

    reason = payload.get("reason")
    note = payload.get("note")
    if reason is not None and not isinstance(reason, str):
        return jsonify({"error": "'reason' must be a string"}), 400
    if note is not None and not isinstance(note, str):
        return jsonify({"error": "'note' must be a string"}), 400

    try:
        result = apply_feedback_action(paper_id, action, reason=reason, note=note)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify(result)


@api_bp.route("/papers/bulk-feedback", methods=["POST"])
def bulk_feedback():
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    paper_ids = payload.get("paper_ids", [])
    action = payload.get("action")
    if not isinstance(paper_ids, list) or not paper_ids:
        return jsonify({"error": "Missing 'paper_ids'"}), 400
    if not isinstance(action, str):
        return jsonify({"error": "Missing 'action'"}), 400

    from app.services.feedback import ALLOWED_ACTIONS

    if action not in ALLOWED_ACTIONS:
        return jsonify({"error": f"Unsupported action '{action}'"}), 400

    results = []
    for pid in paper_ids:
        # Only integer primary keys are valid; a dict/list id reaches SQLAlchemy as a
        # malformed PK and raises InvalidRequestError (not caught below) → 500.
        if not isinstance(pid, int) or isinstance(pid, bool):
            continue
        try:
            result = apply_feedback_action(pid, action)
            results.append(result)
        except (ValueError, LookupError):
            continue
    return jsonify({"processed": len(results), "results": results})


@api_bp.route("/papers/<int:paper_id>/explain", methods=["GET"])
def paper_explain(paper_id: int):
    """Return ranking explanations for a paper."""
    from app.services.ranking import explain_score, generate_ranking_explanation

    paper = db.session.get(Paper, paper_id) or abort(404, description="Paper not found")
    config = current_app.config["SCRAPER_CONFIG"]
    match_types = paper.match_types
    breakdown = explain_score(
        match_types=match_types,
        matched_terms_count=len(paper.matched_terms_list),
        publication_dt=paper.publication_dt,
        resource_count=len(paper.resource_links_list),
        llm_relevance_score=paper.llm_relevance_score,
        citation_count=paper.citation_count,
        acceptance_status=paper.acceptance_status,
        interest_similarity=paper.interest_similarity,
        feedback_score=int(paper.feedback_score or 0),
        config=config,
    )
    explanations = generate_ranking_explanation(paper, config=config)
    return jsonify({"paper_id": paper.id, **breakdown, "explanations": explanations})


@api_bp.route("/papers/<int:paper_id>/follow", methods=["POST"])
def follow_recommendation(paper_id: int):
    validate_csrf_token()
    paper = db.session.get(Paper, paper_id) or abort(404, description="Paper not found")

    term = first_author_name(paper.authors)
    if not term:
        return jsonify({"error": "No author available to follow"}), 400

    with config_write_lock():
        full_config, added = append_whitelist_term(current_app.config["SCRAPER_CONFIG"], "authors", term)
        persist_config(full_config)
    return jsonify({"term": term, "added": added, "message": f"Following {term}."})


@api_bp.route("/papers/<int:paper_id>/mute", methods=["POST"])
def mute_recommendation(paper_id: int):
    validate_csrf_token()
    paper = db.session.get(Paper, paper_id) or abort(404, description="Paper not found")

    term = next((tag for tag in paper.topic_tags_list if tag), "")
    if not term:
        return jsonify({"error": "No topic available to mute"}), 400

    with config_write_lock():
        full_config, added = append_muted_term(current_app.config["SCRAPER_CONFIG"], "topics", term)
        persist_config(full_config)
    return jsonify({"term": term, "added": added, "message": f"Muted topic {term}."})
