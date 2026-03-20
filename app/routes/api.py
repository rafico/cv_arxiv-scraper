from pathlib import Path

from flask import Blueprint, Response, abort, current_app, jsonify, request

from app.csrf import validate_csrf_token
from app.models import Paper, db
from app import _validate_config
from app.services.preferences import (
    append_muted_term,
    append_whitelist_term,
    first_author_name,
    save_config,
)
from app.services import SCRAPE_JOB_MANAGER, apply_feedback_action, stream_or_start_scrape
from app.services.bibtex import paper_to_bibtex, papers_to_bibtex
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


@api_bp.route("/export/bibtex", methods=["GET"])
def export_bibtex():
    from datetime import timedelta
    from app.routes.dashboard import TIMEFRAME_DAYS
    from app.services.ranking import FEEDBACK_BOOST
    from app.services.text import now_utc

    timeframe = request.args.get("timeframe", "daily")
    view = request.args.get("view", "inbox")

    if timeframe not in TIMEFRAME_DAYS:
        timeframe = "daily"

    query = Paper.query.filter(Paper.is_hidden.is_(False))

    if view == "saved":
        from app.models import PaperFeedback
        query = query.join(
            PaperFeedback,
            db.and_(PaperFeedback.paper_id == Paper.id, PaperFeedback.action == "save"),
        )

    days = TIMEFRAME_DAYS.get(timeframe)
    if days is not None:
        cutoff = now_utc() - timedelta(days=days)
        cutoff_date = cutoff.date()
        query = query.filter(
            db.or_(
                Paper.publication_dt >= cutoff_date,
                db.and_(Paper.publication_dt.is_(None), Paper.scraped_at >= cutoff),
            )
        )

    papers = query.order_by(
        (
            db.func.coalesce(Paper.paper_score, 0.0)
            + db.func.coalesce(Paper.feedback_score, 0) * FEEDBACK_BOOST
        ).desc(),
    ).all()

    bib = papers_to_bibtex(papers)
    response = Response(bib, mimetype="application/x-bibtex")
    response.headers["Content-Disposition"] = (
        f'attachment; filename="arxiv_papers_{timeframe}.bib"'
    )
    return response


@api_bp.route("/papers/<int:paper_id>/bibtex", methods=["GET"])
def single_paper_bibtex(paper_id: int):
    paper = db.session.get(Paper, paper_id) or abort(404)
    bib = paper_to_bibtex(paper)
    return Response(bib, mimetype="application/x-bibtex")


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


@api_bp.route("/papers/<int:paper_id>/follow", methods=["POST"])
def follow_recommendation(paper_id: int):
    validate_csrf_token()
    paper = db.session.get(Paper, paper_id) or abort(404)

    term = first_author_name(paper.authors)
    if not term:
        return jsonify({"error": "No author available to follow"}), 400

    config_path = Path(current_app.config["CONFIG_PATH"])
    full_config, added = append_whitelist_term(current_app.config["SCRAPER_CONFIG"], "authors", term)
    _validate_config(full_config, config_path=config_path)
    save_config(config_path, full_config)
    current_app.config["SCRAPER_CONFIG"] = full_config
    return jsonify({"term": term, "added": added, "message": f"Following {term}."})


@api_bp.route("/papers/<int:paper_id>/mute", methods=["POST"])
def mute_recommendation(paper_id: int):
    validate_csrf_token()
    paper = db.session.get(Paper, paper_id) or abort(404)

    term = next((tag for tag in paper.topic_tags_list if tag), "")
    if not term:
        return jsonify({"error": "No topic available to mute"}), 400

    config_path = Path(current_app.config["CONFIG_PATH"])
    full_config, added = append_muted_term(current_app.config["SCRAPER_CONFIG"], "topics", term)
    _validate_config(full_config, config_path=config_path)
    save_config(config_path, full_config)
    current_app.config["SCRAPER_CONFIG"] = full_config
    return jsonify({"term": term, "added": added, "message": f"Muted topic {term}."})
