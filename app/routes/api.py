from flask import Blueprint, Response, current_app, jsonify, request

from app.services import SCRAPE_JOB_MANAGER, apply_feedback_action, stream_or_start_scrape

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/scrape", methods=["POST"])
def trigger_scrape():
    app = current_app._get_current_object()
    job = SCRAPE_JOB_MANAGER.start_or_get_active(app)
    return jsonify(
        {
            "job_id": job.id,
            "status": job.status,
            "started_at": job.started_at.isoformat(),
        }
    )


@api_bp.route("/scrape/status", methods=["GET"])
def scrape_status():
    job_id = SCRAPE_JOB_MANAGER._active_job_id
    if not job_id:
        return jsonify({"running": False})
    job = SCRAPE_JOB_MANAGER._jobs.get(job_id)
    if not job or job.finished_at is not None:
        return jsonify({"running": False})
    return jsonify({"running": True, "status": job.status})


@api_bp.route("/scrape/stream", methods=["GET"])
def scrape_stream():
    app = current_app._get_current_object()
    return Response(
        stream_or_start_scrape(app),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@api_bp.route("/papers/<int:paper_id>/feedback", methods=["POST"])
def paper_feedback(paper_id: int):
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
