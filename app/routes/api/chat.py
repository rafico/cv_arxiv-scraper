"""Conversational RAG endpoints: corpus chat + grounded per-paper chat."""

from flask import abort, jsonify, request
from werkzeug.exceptions import BadRequest

from app.csrf import validate_csrf_token
from app.models import Collection, Paper, db
from app.routes.api import api_bp
from app.routes.api._validation import require_list, require_str
from app.services import rag

# Generous for a question, tight enough to keep prompts (and abuse) bounded.
_MAX_QUESTION_CHARS = 2000


@api_bp.route("/corpus/chat", methods=["POST"])
def corpus_chat():
    """Answer a question grounded in a collection, the saved papers, or (nothing saved) the corpus."""
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    query = require_str(payload, "query")
    if len(query) > _MAX_QUESTION_CHARS:
        raise BadRequest(f"'query' must be at most {_MAX_QUESTION_CHARS} characters")

    paper_ids = None
    collection_id = payload.get("collection_id")
    if collection_id is not None:
        if isinstance(collection_id, bool) or not isinstance(collection_id, int):
            raise BadRequest("'collection_id' must be an integer")
        collection = db.session.get(Collection, collection_id) or abort(404, description="Collection not found")
        paper_ids = [membership.paper_id for membership in collection.papers]

    return jsonify(rag.answer_query(query, paper_ids=paper_ids))


@api_bp.route("/papers/<int:paper_id>/chat", methods=["POST"])
def paper_chat(paper_id: int):
    """Answer a question grounded in one paper's own indexed text (with citations)."""
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    question = require_str(payload, "question")
    if len(question) > _MAX_QUESTION_CHARS:
        raise BadRequest(f"'question' must be at most {_MAX_QUESTION_CHARS} characters")
    history = require_list(payload, "history", default=[])

    if db.session.get(Paper, paper_id) is None:
        abort(404, description="Paper not found")

    from app.services import paper_chat as paper_chat_service

    result = paper_chat_service.answer_paper_question(paper_id, question, history)

    if result.get("llm_error"):
        # The LLM is configured but the upstream call failed — surface it honestly
        # instead of silently degrading to the no-LLM view.
        return jsonify({"error": "The AI provider request failed. Try again or check Settings → AI."}), 502

    return jsonify(result)
