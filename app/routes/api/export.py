"""HTML report and BibTeX export endpoints."""

from flask import Response, abort, current_app, jsonify, request

from app.enums import FeedbackAction
from app.models import Paper, db, inbox_freshness_clause
from app.routes.api import api_bp
from app.services.bibtex import paper_to_bibtex, papers_to_bibtex
from app.services.export import generate_html_report


@api_bp.route("/export", methods=["GET"])
def export_html():
    app = current_app._get_current_object()
    timeframe = request.args.get("timeframe", "daily")
    html = generate_html_report(app, timeframe=timeframe)
    response = current_app.response_class(html, mimetype="text/html")
    if request.args.get("download") == "1":
        response.headers["Content-Disposition"] = f'attachment; filename="arxiv_report_{timeframe}.html"'
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
            db.and_(PaperFeedback.paper_id == Paper.id, PaperFeedback.action == FeedbackAction.SAVE.value),
        )

    days = TIMEFRAME_DAYS.get(timeframe)
    if days is not None:
        cutoff = now_utc() - timedelta(days=days)
        query = query.filter(inbox_freshness_clause(cutoff))

    papers = query.order_by(
        (db.func.coalesce(Paper.paper_score, 0.0) + db.func.coalesce(Paper.feedback_score, 0) * FEEDBACK_BOOST).desc(),
    ).all()

    bib = papers_to_bibtex(papers)
    response = Response(bib, mimetype="application/x-bibtex")
    response.headers["Content-Disposition"] = f'attachment; filename="arxiv_papers_{timeframe}.bib"'
    return response


@api_bp.route("/papers/<int:paper_id>/bibtex", methods=["GET"])
def single_paper_bibtex(paper_id: int):
    paper = db.session.get(Paper, paper_id) or abort(404)
    bib = paper_to_bibtex(paper)
    return Response(bib, mimetype="application/x-bibtex")


@api_bp.route("/papers/bulk-bibtex", methods=["GET"])
def bulk_bibtex():
    ids_param = request.args.get("ids", "")
    try:
        paper_ids = [int(x.strip()) for x in ids_param.split(",") if x.strip()]
    except ValueError:
        return jsonify({"error": "Invalid paper IDs"}), 400
    if not paper_ids:
        return Response("", mimetype="application/x-bibtex")
    papers = Paper.query.filter(Paper.id.in_(paper_ids)).all()
    bib = papers_to_bibtex(papers)
    return Response(bib, mimetype="application/x-bibtex")
