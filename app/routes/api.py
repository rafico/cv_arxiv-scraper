from pathlib import Path

from flask import Blueprint, Response, abort, current_app, jsonify, request

from app.csrf import validate_csrf_token
from app.models import Collection, Paper, PaperCollection, PaperRelation, SavedSearch, db
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


VALID_READING_STATUSES = {"to_read", "reading", "read"}


@api_bp.route("/papers/<int:paper_id>/reading-status", methods=["POST"])
def paper_reading_status(paper_id: int):
    validate_csrf_token()
    paper = db.session.get(Paper, paper_id) or abort(404)
    payload = request.get_json(silent=True) or {}
    status = payload.get("status")
    if status is not None and status not in VALID_READING_STATUSES:
        return jsonify({"error": f"Invalid status. Must be one of: {', '.join(sorted(VALID_READING_STATUSES))}"}), 400
    paper.reading_status = status
    db.session.commit()
    return jsonify({"paper_id": paper.id, "reading_status": paper.reading_status})


@api_bp.route("/papers/<int:paper_id>/notes", methods=["PUT"])
def paper_notes(paper_id: int):
    validate_csrf_token()
    paper = db.session.get(Paper, paper_id) or abort(404)
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
    paper = db.session.get(Paper, paper_id) or abort(404)
    payload = request.get_json(silent=True) or {}
    tag = payload.get("tag", "").strip()
    if not tag:
        return jsonify({"error": "Missing 'tag'"}), 400
    current = list(paper.user_tags or [])
    if tag not in current:
        current.append(tag)
        paper.user_tags = current
        db.session.commit()
    return jsonify({"paper_id": paper.id, "user_tags": paper.user_tags})


@api_bp.route("/papers/<int:paper_id>/tags", methods=["DELETE"])
def paper_remove_tag(paper_id: int):
    validate_csrf_token()
    paper = db.session.get(Paper, paper_id) or abort(404)
    payload = request.get_json(silent=True) or {}
    tag = payload.get("tag", "").strip()
    if not tag:
        return jsonify({"error": "Missing 'tag'"}), 400
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


# ── Collections API ──


@api_bp.route("/collections", methods=["GET"])
def list_collections():
    collections = Collection.query.order_by(Collection.name).all()
    return jsonify([
        {
            "id": c.id,
            "name": c.name,
            "description": c.description or "",
            "color": c.color,
            "paper_count": PaperCollection.query.filter_by(collection_id=c.id).count(),
        }
        for c in collections
    ])


@api_bp.route("/collections", methods=["POST"])
def create_collection():
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Missing 'name'"}), 400
    if Collection.query.filter_by(name=name).first():
        return jsonify({"error": "Collection already exists"}), 409
    c = Collection(
        name=name,
        description=(payload.get("description") or "").strip(),
        color=(payload.get("color") or "").strip() or None,
    )
    db.session.add(c)
    db.session.commit()
    return jsonify({"id": c.id, "name": c.name}), 201


@api_bp.route("/collections/<int:collection_id>", methods=["PUT"])
def update_collection(collection_id: int):
    validate_csrf_token()
    c = db.session.get(Collection, collection_id) or abort(404)
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    if name:
        c.name = name
    if "description" in payload:
        c.description = (payload["description"] or "").strip()
    if "color" in payload:
        c.color = (payload["color"] or "").strip() or None
    db.session.commit()
    return jsonify({"id": c.id, "name": c.name})


@api_bp.route("/collections/<int:collection_id>", methods=["DELETE"])
def delete_collection(collection_id: int):
    validate_csrf_token()
    c = db.session.get(Collection, collection_id) or abort(404)
    db.session.delete(c)
    db.session.commit()
    return jsonify({"deleted": True})


@api_bp.route("/collections/<int:collection_id>/papers", methods=["POST"])
def add_paper_to_collection(collection_id: int):
    validate_csrf_token()
    c = db.session.get(Collection, collection_id) or abort(404)
    payload = request.get_json(silent=True) or {}
    paper_ids = payload.get("paper_ids", [])
    if isinstance(payload.get("paper_id"), int):
        paper_ids = [payload["paper_id"]]
    added = 0
    for pid in paper_ids:
        if not db.session.get(Paper, pid):
            continue
        if PaperCollection.query.filter_by(paper_id=pid, collection_id=c.id).first():
            continue
        db.session.add(PaperCollection(paper_id=pid, collection_id=c.id))
        added += 1
    db.session.commit()
    return jsonify({"added": added, "collection_id": c.id})


@api_bp.route("/collections/<int:collection_id>/papers/<int:paper_id>", methods=["DELETE"])
def remove_paper_from_collection(collection_id: int, paper_id: int):
    validate_csrf_token()
    pc = PaperCollection.query.filter_by(paper_id=paper_id, collection_id=collection_id).first()
    if not pc:
        abort(404)
    db.session.delete(pc)
    db.session.commit()
    return jsonify({"removed": True})


# ── Saved Searches API ──


@api_bp.route("/saved-searches", methods=["GET"])
def list_saved_searches():
    searches = SavedSearch.query.order_by(SavedSearch.created_at.desc()).all()
    return jsonify([
        {"id": s.id, "name": s.name, "filters": s.filters, "created_at": s.created_at.isoformat() if s.created_at else None}
        for s in searches
    ])


@api_bp.route("/saved-searches", methods=["POST"])
def create_saved_search():
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Missing 'name'"}), 400
    filters = payload.get("filters", {})
    if not isinstance(filters, dict):
        return jsonify({"error": "'filters' must be a dict"}), 400
    s = SavedSearch(name=name, filters=filters)
    db.session.add(s)
    db.session.commit()
    return jsonify({"id": s.id, "name": s.name}), 201


@api_bp.route("/saved-searches/<int:search_id>", methods=["DELETE"])
def delete_saved_search(search_id: int):
    validate_csrf_token()
    s = db.session.get(SavedSearch, search_id) or abort(404)
    db.session.delete(s)
    db.session.commit()
    return jsonify({"deleted": True})


# ── Bulk Operations API ──


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

    results = []
    for pid in paper_ids:
        try:
            result = apply_feedback_action(pid, action)
            results.append(result)
        except (ValueError, LookupError):
            continue
    return jsonify({"processed": len(results), "results": results})


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


# ── Author Search API ──


@api_bp.route("/authors", methods=["GET"])
def search_authors():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    # Extract unique author names matching the query.
    escaped_q = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    rows = (
        db.session.query(Paper.authors)
        .filter(Paper.authors.ilike(f"%{escaped_q}%", escape="\\"))
        .limit(200)
        .all()
    )
    seen: dict[str, int] = {}
    for (authors_str,) in rows:
        for author in (a.strip() for a in authors_str.split(",") if a.strip()):
            if q.lower() in author.lower():
                key = author.strip()
                seen[key] = seen.get(key, 0) + 1

    results = sorted(seen.items(), key=lambda x: (-x[1], x[0]))[:20]
    return jsonify([{"name": name, "paper_count": count} for name, count in results])


# ── Citation/Similarity Graph API ──


@api_bp.route("/papers/<int:paper_id>/graph", methods=["GET"])
def paper_graph(paper_id: int):
    from app.services.related import build_vector, cosine_similarity

    paper = db.session.get(Paper, paper_id) or abort(404)

    # Build graph from top-N similar papers.
    pool = (
        Paper.query
        .filter(Paper.id != paper_id)
        .order_by(Paper.paper_score.desc())
        .limit(100)
        .all()
    )
    center_text = " ".join([paper.title or "", paper.summary_text or "", paper.abstract_text or ""])
    center_vec = build_vector(center_text)

    nodes = [{"id": paper.id, "title": paper.title, "score": float(paper.paper_score or 0), "center": True}]
    edges = []

    for other in pool:
        other_text = " ".join([other.title or "", other.summary_text or "", other.abstract_text or ""])
        other_vec = build_vector(other_text)
        sim = cosine_similarity(center_vec, other_vec)
        if sim >= 0.15:
            nodes.append({"id": other.id, "title": other.title, "score": float(other.paper_score or 0), "center": False})
            edges.append({"source": paper.id, "target": other.id, "similarity": round(sim, 3)})

    # Sort edges by similarity and keep top 20.
    edges.sort(key=lambda e: e["similarity"], reverse=True)
    edges = edges[:20]
    connected_ids = {paper.id}
    for e in edges:
        connected_ids.add(e["target"])
    nodes = [n for n in nodes if n["id"] in connected_ids]

    return jsonify({"nodes": nodes, "edges": edges})


# ── Feed Sources API ──


@api_bp.route("/feed-sources", methods=["GET"])
def list_feed_sources():
    from app.models import FeedSource
    sources = FeedSource.query.order_by(FeedSource.created_at).all()
    return jsonify([
        {
            "id": s.id,
            "name": s.name,
            "url": s.url,
            "feed_type": s.feed_type,
            "enabled": s.enabled,
            "last_fetched_at": s.last_fetched_at.isoformat() if s.last_fetched_at else None,
        }
        for s in sources
    ])


@api_bp.route("/feed-sources", methods=["POST"])
def create_feed_source():
    from app.models import FeedSource
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    url = (payload.get("url") or "").strip()
    if not name or not url:
        return jsonify({"error": "Missing 'name' or 'url'"}), 400
    feed_type = payload.get("feed_type", "arxiv_rss")
    s = FeedSource(name=name, url=url, feed_type=feed_type)
    db.session.add(s)
    db.session.commit()
    return jsonify({"id": s.id, "name": s.name}), 201


@api_bp.route("/feed-sources/<int:source_id>", methods=["DELETE"])
def delete_feed_source(source_id: int):
    from app.models import FeedSource
    validate_csrf_token()
    s = db.session.get(FeedSource, source_id) or abort(404)
    db.session.delete(s)
    db.session.commit()
    return jsonify({"deleted": True})


@api_bp.route("/feed-sources/<int:source_id>/toggle", methods=["POST"])
def toggle_feed_source(source_id: int):
    from app.models import FeedSource
    validate_csrf_token()
    s = db.session.get(FeedSource, source_id) or abort(404)
    s.enabled = not s.enabled
    db.session.commit()
    return jsonify({"id": s.id, "enabled": s.enabled})
