"""Local search, corpus analysis, author search, and similarity graph endpoints."""

from flask import abort, current_app, jsonify, request

from app.models import Collection, Paper, PaperCollection, db
from app.routes.api import api_bp
from app.routes.api._validation import parse_int_query_arg as _parse_int_query_arg


def _parse_bool_query_arg(name: str, *, default: bool) -> bool:
    raw = request.args.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid '{name}' parameter")


def _parse_paper_ids_query(raw: str) -> list[int]:
    if not raw.strip():
        return []
    try:
        return [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("Invalid 'paper_ids' parameter") from exc


@api_bp.route("/search", methods=["GET"])
def search_papers():
    """Local hybrid search endpoint."""
    from app.services.search import search_bm25, search_hybrid, search_semantic

    q = request.args.get("q", "").strip()
    mode = request.args.get("mode", "hybrid")
    try:
        # Lower-bound the cap: a negative LIMIT is "unlimited" in SQLite, which would
        # bypass the 100-result ceiling on the FTS/hybrid query.
        top_k = max(1, min(int(request.args.get("limit", 30)), 100))
    except (ValueError, TypeError):
        top_k = 30

    if not q:
        return jsonify({"query": "", "mode": mode, "results": []})

    if mode == "keyword":
        raw = search_bm25(q, limit=top_k)
        results = [{"paper_id": pid, "score": score} for pid, score in raw]
    elif mode == "semantic":
        raw = search_semantic(q, top_k=top_k)
        results = [{"paper_id": pid, "score": score} for pid, score in raw]
    else:
        results = search_hybrid(q, top_k=top_k)

    # Enrich with paper data
    paper_ids = [r["paper_id"] for r in results]
    papers_by_id = {p.id: p for p in Paper.query.filter(Paper.id.in_(paper_ids)).all()} if paper_ids else {}

    enriched = []
    for r in results:
        paper = papers_by_id.get(r["paper_id"])
        if paper:
            enriched.append(
                {
                    **r,
                    "title": paper.title,
                    "authors": paper.authors,
                    "arxiv_id": paper.arxiv_id,
                    "abstract": paper.abstract_text[:300] if paper.abstract_text else "",
                }
            )

    return jsonify({"query": q, "mode": mode, "results": enriched})


@api_bp.route("/corpus/clusters", methods=["GET"])
def corpus_clusters():
    from app.services.corpus_analysis import analyze_topic_clusters

    try:
        window_days = _parse_int_query_arg("window_days", default=7, minimum=1, maximum=365)
        offset_days = _parse_int_query_arg("offset_days", default=0, minimum=0, maximum=365)
        limit = _parse_int_query_arg("limit", default=200, minimum=1, maximum=1000)
        cluster_count = _parse_int_query_arg("clusters", default=None, minimum=1, maximum=25)
        paper_limit = _parse_int_query_arg("paper_limit", default=5, minimum=1, maximum=50)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    result = analyze_topic_clusters(
        window_days=window_days,
        offset_days=offset_days,
        limit=limit,
        cluster_count=cluster_count,
        paper_limit=paper_limit,
    )
    return jsonify(result)


@api_bp.route("/corpus/emerging", methods=["GET"])
def corpus_emerging():
    from app.services.corpus_analysis import detect_emerging_topics

    try:
        recent_days = _parse_int_query_arg("recent_days", default=7, minimum=1, maximum=365)
        baseline_days = _parse_int_query_arg("baseline_days", default=28, minimum=1, maximum=3650)
        limit = _parse_int_query_arg("limit", default=200, minimum=1, maximum=1000)
        cluster_count = _parse_int_query_arg("clusters", default=None, minimum=1, maximum=25)
        paper_limit = _parse_int_query_arg("paper_limit", default=3, minimum=1, maximum=50)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    result = detect_emerging_topics(
        recent_days=recent_days,
        baseline_days=baseline_days,
        limit=limit,
        cluster_count=cluster_count,
        paper_limit=paper_limit,
    )
    return jsonify(result)


@api_bp.route("/corpus/neighbors", methods=["GET"])
def corpus_neighbors():
    from app.services.corpus_analysis import find_neighbor_papers

    try:
        limit = _parse_int_query_arg("limit", default=20, minimum=1, maximum=100)
        collection_id = _parse_int_query_arg("collection_id", default=None, minimum=1)
        exclude_tracked_authors = _parse_bool_query_arg("exclude_tracked_authors", default=True)
        seed_paper_ids = _parse_paper_ids_query(request.args.get("paper_ids", ""))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if collection_id is not None:
        collection = db.session.get(Collection, collection_id) or abort(404)
        seed_paper_ids.extend(
            paper_collection.paper_id
            for paper_collection in PaperCollection.query.filter_by(collection_id=collection.id)
            .order_by(PaperCollection.added_at.desc())
            .all()
        )

    if not seed_paper_ids:
        return jsonify({"error": "Provide 'paper_ids' or 'collection_id'"}), 400

    tracked_authors = current_app.config.get("SCRAPER_CONFIG", {}).get("whitelists", {}).get("authors", [])
    result = find_neighbor_papers(
        seed_paper_ids,
        limit=limit,
        tracked_authors=tracked_authors,
        exclude_tracked_authors=exclude_tracked_authors,
    )
    if collection_id is not None:
        result["collection_id"] = collection_id
    return jsonify(result)


@api_bp.route("/authors", methods=["GET"])
def search_authors():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    # Extract unique author names matching the query.
    escaped_q = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    rows = db.session.query(Paper.authors).filter(Paper.authors.ilike(f"%{escaped_q}%", escape="\\")).limit(200).all()
    seen: dict[str, int] = {}
    for (authors_str,) in rows:
        for author in (a.strip() for a in authors_str.split(",") if a.strip()):
            if q.lower() in author.lower():
                key = author.strip()
                seen[key] = seen.get(key, 0) + 1

    results = sorted(seen.items(), key=lambda x: (-x[1], x[0]))[:20]
    return jsonify([{"name": name, "paper_count": count} for name, count in results])


GRAPH_DEFAULT_LIMIT = 500
GRAPH_MAX_LIMIT = 2000


@api_bp.route("/graph", methods=["GET"])
def citation_graph():
    """Citation graph over the corpus (or one collection): nodes + "cites" edges + PageRank."""
    from app.models import PaperRelation
    from app.services.citation_graph import CITES, pagerank

    try:
        limit = _parse_int_query_arg("limit", default=GRAPH_DEFAULT_LIMIT, minimum=1, maximum=GRAPH_MAX_LIMIT)
        collection_id = _parse_int_query_arg("collection", default=None, minimum=1)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    query = Paper.query.filter(Paper.is_hidden.is_(False))
    if collection_id is not None:
        db.session.get(Collection, collection_id) or abort(404)
        query = query.join(
            PaperCollection,
            db.and_(PaperCollection.paper_id == Paper.id, PaperCollection.collection_id == collection_id),
        )
    # Most-cited first so a capped corpus view keeps the interesting nodes.
    papers = query.order_by(Paper.citation_count.desc().nulls_last(), Paper.id).limit(limit).all()

    id_set = {p.id for p in papers}
    # Filter edges in Python rather than two big IN clauses — the "cites" table
    # is small (edges only exist between locally-tracked papers), and 2000-id
    # IN lists flirt with SQLite's variable limit.
    edges = [
        (citing, cited)
        for citing, cited in db.session.query(PaperRelation.paper_id, PaperRelation.related_paper_id).filter(
            PaperRelation.relation_type == CITES
        )
        if citing in id_set and cited in id_set
    ]
    ranks = pagerank([p.id for p in papers], edges)

    nodes = [
        {
            "id": p.id,
            "title": p.title,
            "year": p.publication_dt.year if p.publication_dt else None,
            "citations": p.citation_count or 0,
            "pagerank": round(ranks.get(p.id, 0.0), 6),
            "tags": p.user_tags or [],
        }
        for p in papers
    ]
    return jsonify({"nodes": nodes, "edges": [{"source": a, "target": b} for a, b in edges]})
