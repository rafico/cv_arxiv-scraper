"""Scrape trigger/status/stream and historical search endpoints."""

from flask import Response, current_app, jsonify, request

from app.csrf import validate_csrf_token
from app.routes.api import api_bp
from app.services import SCRAPE_JOB_MANAGER


@api_bp.route("/scrape", methods=["POST"])
def trigger_scrape():
    validate_csrf_token()
    app = current_app._get_current_object()
    payload = request.get_json(silent=True) or {}
    force = bool(payload.get("force"))
    job = SCRAPE_JOB_MANAGER.start_or_get_active(app, force=force)
    return jsonify(
        {
            "job_id": job.id,
            "status": job.status,
            "started_at": job.started_at.isoformat(),
        }
    )


@api_bp.route("/scrape/status", methods=["GET"])
def scrape_status():
    return jsonify(SCRAPE_JOB_MANAGER.get_status_snapshot())


@api_bp.route("/scrape/stream", methods=["GET"])
def scrape_stream():
    # No CSRF validation: this is a read-only SSE endpoint and EventSource
    # cannot send custom headers. Scrape initiation happens on the POST route.
    job_id = request.args.get("job_id", "").strip()
    if not job_id:
        return jsonify({"error": "Missing 'job_id' parameter"}), 400

    return Response(
        SCRAPE_JOB_MANAGER.stream_for_job(job_id),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@api_bp.route("/search/historical", methods=["POST"])
def search_historical():
    from datetime import datetime

    from app.services.scrape_engine import execute_historical_scrape

    validate_csrf_token()
    payload = request.get_json(silent=True) or {}

    raw_categories = payload.get("categories", ["cs.CV"])
    # Require a list of non-empty strings. Validate BEFORE the scrape so a
    # wrong-typed value yields a clean 400 instead of a garbage query (a bare
    # string iterated char-by-char) or a misleading 502 (a non-iterable raising
    # TypeError inside the engine). An empty list normalizes to the default.
    if not isinstance(raw_categories, list):
        return jsonify({"error": "categories must be a list of non-empty strings"}), 400
    categories = [c.strip() for c in raw_categories if isinstance(c, str) and c.strip()]
    if len(categories) != len(raw_categories):
        return jsonify({"error": "categories must be a list of non-empty strings"}), 400
    if not categories:
        categories = ["cs.CV"]

    start_date_str = payload.get("start_date")
    end_date_str = payload.get("end_date")

    if not start_date_str or not end_date_str:
        return jsonify({"error": "start_date and end_date are required"}), 400

    try:
        # A non-string date (e.g. an int/list) makes strptime raise TypeError, not
        # ValueError; catch both so it yields the specific format error, not a
        # generic safety-net 400 with a logged traceback.
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return jsonify({"error": "Dates must be in YYYY-MM-DD format"}), 400

    app = current_app._get_current_object()
    try:
        summary = execute_historical_scrape(app, categories, start_dt, end_dt)
    except Exception:
        current_app.logger.exception("Historical scrape failed")
        return jsonify(
            {"error": "Historical search failed. The arXiv API may be temporarily unavailable; please try again."}
        ), 502
    return jsonify(summary)
