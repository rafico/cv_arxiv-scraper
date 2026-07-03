"""Conversational RAG endpoints: corpus chat + grounded per-paper chat."""

from flask import abort, jsonify, request
from werkzeug.exceptions import BadRequest

from app.csrf import validate_csrf_token
from app.models import Paper, db
from app.routes.api import api_bp
from app.routes.api._validation import require_list, require_str
from app.services import rag

# Generous for a question, tight enough to keep prompts (and abuse) bounded.
_MAX_QUESTION_CHARS = 2000


@api_bp.route("/corpus/chat", methods=["POST"])
def corpus_chat():
    """Answer a question grounded in the reader's saved papers."""
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Missing 'query'"}), 400

    result = rag.answer_query(query.strip())

    if result["no_saved_papers"]:
        # Not an error: the reader simply hasn't saved any papers yet.
        return jsonify(
            {
                "query": result["query"],
                "synthesis": None,
                "llm_used": False,
                "sources": [],
                "no_saved_papers": True,
                "verifications": None,
                "message": "Save some papers first, then chat over your corpus.",
            }
        )

    return jsonify(result)


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
