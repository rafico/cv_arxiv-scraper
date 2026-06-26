"""Conversational RAG endpoint: chat over the reader's saved corpus."""

from flask import jsonify, request

from app.csrf import validate_csrf_token
from app.routes.api import api_bp
from app.services import rag


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
                "message": "Save some papers first, then chat over your corpus.",
            }
        )

    return jsonify(result)
