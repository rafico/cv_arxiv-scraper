from flask import Blueprint, jsonify, Response, current_app
from app.scraper import run_scrape, run_scrape_stream

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/scrape", methods=["POST"])
def trigger_scrape():
    result = run_scrape(current_app._get_current_object())
    return jsonify(result)


@api_bp.route("/scrape/stream", methods=["GET"])
def scrape_stream():
    app = current_app._get_current_object()
    return Response(
        run_scrape_stream(app),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
