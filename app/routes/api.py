from flask import Blueprint, Response, current_app, jsonify, request

from app.csrf import validate_csrf_token
from app.services import SCRAPE_JOB_MANAGER, apply_feedback_action, stream_or_start_scrape
from app.services.export import generate_html_report

api_bp = Blueprint("api", __name__, url_prefix="/api")


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
    validate_csrf_token()
    app = current_app._get_current_object()
    force = request.args.get("force", "").strip().lower() in {"1", "true", "yes", "on"}
    return Response(
        stream_or_start_scrape(app, force=force),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@api_bp.route("/export", methods=["GET"])
def export_html():
    app = current_app._get_current_object()
    timeframe = request.args.get("timeframe", "daily")
    html = generate_html_report(app, timeframe=timeframe)
    response = current_app.response_class(html, mimetype="text/html")
    if request.args.get("download") == "1":
        response.headers["Content-Disposition"] = (
            f'attachment; filename="arxiv_report_{timeframe}.html"'
        )
    return response


@api_bp.route("/papers/<int:paper_id>/feedback", methods=["POST"])
def paper_feedback(paper_id: int):
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    action = payload.get("action")
    if not isinstance(action, str):
        return jsonify({"error": "Missing 'action'"}), 400

    try:
        result = apply_feedback_action(paper_id, action)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify(result)
