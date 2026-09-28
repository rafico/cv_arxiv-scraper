"""Collections CRUD and membership endpoints."""

from flask import Response, abort, jsonify, request
from sqlalchemy.exc import IntegrityError

from app.csrf import validate_csrf_token
from app.models import Collection, Paper, PaperCollection, db
from app.routes.api import api_bp
from app.routes.api._validation import optional_str, require_list, require_str


@api_bp.route("/collections", methods=["GET"])
def list_collections():
    paper_count_subquery = (
        db.session.query(
            PaperCollection.collection_id,
            db.func.count(PaperCollection.id).label("paper_count"),
        )
        .group_by(PaperCollection.collection_id)
        .subquery()
    )
    results = (
        db.session.query(Collection, db.func.coalesce(paper_count_subquery.c.paper_count, 0))
        .outerjoin(paper_count_subquery, Collection.id == paper_count_subquery.c.collection_id)
        .order_by(Collection.name)
        .all()
    )
    return jsonify(
        [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description or "",
                "color": c.color,
                "paper_count": count,
            }
            for c, count in results
        ]
    )


@api_bp.route("/collections", methods=["POST"])
def create_collection():
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    name = require_str(payload, "name")
    if Collection.query.filter_by(name=name).first():
        return jsonify({"error": "Collection already exists"}), 409
    c = Collection(
        name=name,
        description=optional_str(payload, "description"),
        color=optional_str(payload, "color") or None,
    )
    db.session.add(c)
    db.session.commit()
    return jsonify({"id": c.id, "name": c.name}), 201


@api_bp.route("/collections/<int:collection_id>", methods=["PUT"])
def update_collection(collection_id: int):
    validate_csrf_token()
    c = db.session.get(Collection, collection_id) or abort(404)
    payload = request.get_json(silent=True) or {}
    name = optional_str(payload, "name")
    if name:
        if Collection.query.filter(Collection.name == name, Collection.id != c.id).first():
            return jsonify({"error": "Collection already exists"}), 409
        c.name = name
    if "description" in payload:
        c.description = optional_str(payload, "description")
    if "color" in payload:
        c.color = optional_str(payload, "color") or None
    try:
        db.session.commit()
    except IntegrityError:
        # Closes the (single-worker, narrow) race between the pre-check and commit.
        db.session.rollback()
        return jsonify({"error": "Collection already exists"}), 409
    return jsonify({"id": c.id, "name": c.name})


@api_bp.route("/collections/<int:collection_id>", methods=["DELETE"])
def delete_collection(collection_id: int):
    validate_csrf_token()
    c = db.session.get(Collection, collection_id) or abort(404)
    db.session.delete(c)
    db.session.commit()
    return jsonify({"deleted": True})


@api_bp.route("/collections/<int:collection_id>/export", methods=["GET"])
def export_collection_bundle(collection_id: int):
    from app.services.collection_share import export_collection

    db.session.get(Collection, collection_id) or abort(404)
    response = jsonify(export_collection(collection_id))
    response.headers["Content-Disposition"] = f'attachment; filename="collection-{collection_id}.json"'
    return response


_CSV_COLUMNS = (
    "arxiv_id",
    "title",
    "first_author",
    "year",
    "venue",
    "acceptance_status",
    "github_repo",
    "github_stars",
    "citation_count",
    "readiness",
    "user_tags",
    "user_notes",
    "reading_status",
    "link",
)


def _csv_cell(value):
    # Titles/notes are untrusted text: a spreadsheet evaluates a cell starting with
    # = + - @ (or tab/CR) as a formula, so prefix an apostrophe to keep it literal.
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


@api_bp.route("/collections/<int:collection_id>/table.csv", methods=["GET"])
def export_collection_csv(collection_id: int):
    """One row per paper: the screening/extraction spreadsheet for a review."""
    import csv
    import io

    from app.services.implementation_readiness import implementation_readiness
    from app.services.preferences import first_author_name

    db.session.get(Collection, collection_id) or abort(404)
    papers = (
        Paper.query.join(PaperCollection, PaperCollection.paper_id == Paper.id)
        .filter(PaperCollection.collection_id == collection_id, Paper.is_hidden.is_(False))
        .order_by(Paper.id)
        .all()
    )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_CSV_COLUMNS)
    for p in papers:
        row = (
            p.arxiv_id,
            p.title,
            first_author_name(p.authors),
            p.publication_dt.year if p.publication_dt else None,
            p.venue,
            p.acceptance_status,
            p.github_repo,
            p.github_stars,
            p.citation_count if p.citation_count is not None else p.openalex_cited_by_count,
            implementation_readiness(p).tier,
            "; ".join(p.user_tags_list),
            p.user_notes,
            p.reading_status,
            p.link,
        )
        writer.writerow([_csv_cell(v) for v in row])
    response = Response(buf.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = f'attachment; filename="collection-{collection_id}.csv"'
    return response


_MAX_BUNDLE_UPLOAD_BYTES = 64 * 1024 * 1024
# ponytail: the arXiv fetch and CPU embedding run synchronously on the single
# worker, so seeds are capped and bigger bundles skip embedding (left to
# cv-arxiv-backfill embeddings); a background job if reviews need bigger seeds.
_MAX_IMPORT_IDS = 100


@api_bp.route("/collections/import", methods=["POST"])
def import_collection_bundle():
    from app.services.collection_share import import_collection

    validate_csrf_token()
    request.max_content_length = _MAX_BUNDLE_UPLOAD_BYTES
    manifest = request.get_json(silent=True)
    try:
        collection, stats = import_collection(manifest, embed_max=_MAX_IMPORT_IDS)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"id": collection.id, "name": collection.name, **stats}), 201


@api_bp.route("/collections/import-ids", methods=["POST"])
def import_collection_ids():
    """Seed a collection from arXiv ids/URLs/.bib text pasted as ``text``.

    ``name`` is a collection name (created on miss) or a numeric id. Unlike the
    onboarding bootstrap, no feedback rows are written, so the ranker is untouched.
    """
    from app.services.collection_share import BUNDLE_VERSION, arxiv_bundle_entry, import_collection
    from app.services.mcp_tools import _resolve_or_create_collection
    from app.services.onboarding import extract_arxiv_ids, fetch_arxiv_metadata

    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    name = require_str(payload, "name")
    ids = extract_arxiv_ids(require_str(payload, "text"))
    if not ids:
        return jsonify({"error": "No arXiv ids found"}), 400
    if len(ids) > _MAX_IMPORT_IDS:
        return jsonify({"error": f"Too many arXiv ids (max {_MAX_IMPORT_IDS})"}), 400
    collection, _created = _resolve_or_create_collection(name)
    if collection is None:
        abort(404)

    local = {p.arxiv_id: p for p in Paper.query.filter(Paper.arxiv_id.in_(ids))}
    # Hidden papers link but stay out of the collection view/.bib/MCP; report them.
    hidden = [aid for aid, p in local.items() if p.is_hidden]
    missing = [aid for aid in ids if aid not in local]
    try:
        fetched = fetch_arxiv_metadata(missing) if missing else []
    except Exception:
        return jsonify({"error": "arXiv fetch failed; try again later"}), 502
    papers = [{"arxiv_id": p.arxiv_id, "title": p.title, "link": p.link} for p in local.values()]
    papers += [
        arxiv_bundle_entry(
            e["arxiv_id"], e["title"], e["authors"], e["abstract"], e["publication_date"], e["categories"]
        )
        for e in fetched
    ]
    manifest = {"bundle_version": BUNDLE_VERSION, "collection": {"name": collection.name}, "papers": papers}
    _collection, stats = import_collection(manifest, into=collection)
    found = {entry["arxiv_id"] for entry in papers}
    return jsonify(
        {
            "collection_id": collection.id,
            **stats,
            "not_found": [aid for aid in ids if aid not in found],
            "hidden": hidden,
        }
    )


@api_bp.route("/collections/<int:collection_id>/papers", methods=["POST"])
def add_paper_to_collection(collection_id: int):
    validate_csrf_token()
    c = db.session.get(Collection, collection_id) or abort(404)
    payload = request.get_json(silent=True) or {}
    # ``bool`` is a subclass of ``int``: a JSON ``true``/``false`` would otherwise
    # resolve to ``Paper`` id 1/0 and silently add the wrong paper (mirrors the
    # bulk_feedback guard).
    if isinstance(payload.get("paper_id"), int) and not isinstance(payload.get("paper_id"), bool):
        paper_ids = [payload["paper_id"]]
    else:
        paper_ids = require_list(payload, "paper_ids")
    added = _stage_new_memberships(paper_ids, c.id)
    try:
        db.session.commit()
    except IntegrityError:
        # A concurrent request already inserted one of these rows (the narrow
        # pre-check/commit race under multiple threads); roll back and re-run
        # the toggle so the now-present rows are treated as no-ops and the
        # end-state (papers in collection) is reached idempotently, mirroring
        # create_collection/update_collection instead of an opaque 500.
        db.session.rollback()
        added = _stage_new_memberships(paper_ids, c.id)
        db.session.commit()
    return jsonify({"added": added, "collection_id": c.id})


def _stage_new_memberships(paper_ids: list, collection_id: int) -> int:
    """Stage PaperCollection rows for papers not already in the collection.

    Returns the count staged (not yet committed); skips non-int/bool ids,
    missing papers, and existing memberships."""
    added = 0
    for pid in paper_ids:
        if not isinstance(pid, int) or isinstance(pid, bool) or not db.session.get(Paper, pid):
            continue
        if PaperCollection.query.filter_by(paper_id=pid, collection_id=collection_id).first():
            continue
        db.session.add(PaperCollection(paper_id=pid, collection_id=collection_id))
        added += 1
    return added


@api_bp.route("/collections/<int:collection_id>/papers/<int:paper_id>", methods=["DELETE"])
def remove_paper_from_collection(collection_id: int, paper_id: int):
    validate_csrf_token()
    pc = PaperCollection.query.filter_by(paper_id=paper_id, collection_id=collection_id).first()
    if not pc:
        abort(404)
    db.session.delete(pc)
    db.session.commit()
    return jsonify({"removed": True})
