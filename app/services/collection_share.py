"""Portable collection bundles: export one collection (papers + notes + tags
+ reference ids) as plain JSON, and import such a bundle as a new collection.

No PDFs and no archive format on purpose — papers are re-fetchable from
arXiv, so a bundle is a single JSON document and the whole zip-hardening
surface disappears. Citation edges are not shipped either: each paper's
referenced_works travels with it, so sync_citation_edges() reproduces the
edges exactly on the receiving side.
"""

from __future__ import annotations

BUNDLE_VERSION = 1
MAX_BUNDLE_PAPERS = 5000
MAX_FIELD_CHARS = 200_000  # sanity cap on any single string field

_PAPER_FIELDS = (
    "arxiv_id",
    "title",
    "authors",
    "link",
    "pdf_link",
    "abstract_text",
    "summary_text",
    "publication_date",
    "venue",
    "categories",
    "citation_count",
    "user_notes",
    "user_tags",
    "openalex_id",
    "semantic_scholar_id",
    "referenced_works",
)


def export_collection(collection_id: int) -> dict:
    from app.models import Collection, Paper, PaperAnnotation, PaperCollection, db

    collection = db.session.get(Collection, collection_id)
    if collection is None:
        raise ValueError("Collection not found")
    papers = (
        Paper.query.join(PaperCollection, PaperCollection.paper_id == Paper.id)
        .filter(PaperCollection.collection_id == collection.id)
        .order_by(Paper.id)
        .all()
    )
    index_by_id = {p.id: i for i, p in enumerate(papers)}
    annotations = (
        PaperAnnotation.query.filter(PaperAnnotation.paper_id.in_(index_by_id)).order_by(PaperAnnotation.id).all()
        if index_by_id
        else []
    )
    return {
        "bundle_version": BUNDLE_VERSION,
        "collection": {
            "name": collection.name,
            "description": collection.description or "",
            "color": collection.color,
        },
        "papers": [{field: getattr(p, field) for field in _PAPER_FIELDS} for p in papers],
        "annotations": [
            {
                "paper": index_by_id[a.paper_id],
                "page": a.page,
                "kind": a.kind,
                "rects": a.rects,
                "color": a.color,
                "note": a.note,
            }
            for a in annotations
        ],
    }


def _unique_collection_name(base_name: str) -> str:
    from app.models import Collection

    name = base_name
    suffix = 1
    while Collection.query.filter_by(name=name).first():
        suffix += 1
        name = f"{base_name} (imported)" if suffix == 2 else f"{base_name} (imported {suffix - 1})"
    return name


def _validate(manifest: object) -> dict:
    if not isinstance(manifest, dict) or manifest.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError("Not a collection bundle (missing or unsupported bundle_version)")
    collection = manifest.get("collection")
    if not isinstance(collection, dict) or not isinstance(collection.get("name"), str) or not collection["name"].strip():
        raise ValueError("Bundle has no collection name")
    papers = manifest.get("papers")
    if not isinstance(papers, list):
        raise ValueError("Bundle has no papers list")
    if len(papers) > MAX_BUNDLE_PAPERS:
        raise ValueError(f"Bundle too large (max {MAX_BUNDLE_PAPERS} papers)")
    for entry in papers:
        if not isinstance(entry, dict):
            raise ValueError("Malformed paper entry")
        for field in ("title", "link"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f"Paper entry missing '{field}'")
        for field, value in entry.items():
            if isinstance(value, str) and len(value) > MAX_FIELD_CHARS:
                raise ValueError(f"Paper field '{field}' too long")
    return manifest


def _entry_str(entry: dict, field: str, default: str = "") -> str:
    value = entry.get(field)
    return value if isinstance(value, str) else default


def _entry_list(entry: dict, field: str) -> list:
    value = entry.get(field)
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def import_collection(manifest: object):
    """Import a bundle as a new collection; returns (collection, stats dict).

    Papers are global here (unlike per-dataset stores): an entry matching a
    local paper (by arxiv_id, then link) links into the new collection and
    only fills user_notes/user_tags where locally empty — imports never
    overwrite local data. Unmatched entries become new Paper rows.
    """
    from datetime import date

    from app.models import Collection, Paper, PaperAnnotation, PaperCollection, db
    from app.services.citation_graph import sync_citation_edges

    manifest = _validate(manifest)

    collection = Collection(
        name=_unique_collection_name(manifest["collection"]["name"].strip()),
        description=str(manifest["collection"].get("description") or ""),
        color=manifest["collection"].get("color") if isinstance(manifest["collection"].get("color"), str) else None,
    )
    db.session.add(collection)
    db.session.flush()

    imported_papers: list = []
    linked = created = 0
    for entry in manifest["papers"]:
        arxiv_id = entry.get("arxiv_id") if isinstance(entry.get("arxiv_id"), str) else None
        paper = Paper.query.filter_by(arxiv_id=arxiv_id).first() if arxiv_id else None
        if paper is None:
            paper = Paper.query.filter_by(link=entry["link"]).first()

        if paper is None:
            publication_date = _entry_str(entry, "publication_date") or None
            try:
                publication_dt = date.fromisoformat(publication_date[:10]) if publication_date else None
            except ValueError:
                publication_dt = None
            paper = Paper(
                arxiv_id=arxiv_id,
                title=entry["title"],
                authors=_entry_str(entry, "authors", "Unknown"),
                link=entry["link"],
                pdf_link=_entry_str(entry, "pdf_link") or entry["link"],
                abstract_text=_entry_str(entry, "abstract_text"),
                summary_text=_entry_str(entry, "summary_text"),
                publication_date=publication_date,
                publication_dt=publication_dt,
                venue=_entry_str(entry, "venue") or None,
                categories=_entry_list(entry, "categories"),
                citation_count=entry.get("citation_count") if isinstance(entry.get("citation_count"), int) else None,
                user_notes=_entry_str(entry, "user_notes"),
                user_tags=_entry_list(entry, "user_tags"),
                openalex_id=_entry_str(entry, "openalex_id") or None,
                semantic_scholar_id=_entry_str(entry, "semantic_scholar_id") or None,
                referenced_works=_entry_list(entry, "referenced_works"),
                match_type="import",
                scraped_date=date.today().isoformat(),
            )
            db.session.add(paper)
            db.session.flush()
            created += 1
        else:
            # Fill-only merge: never overwrite local data with imported data.
            if not paper.user_notes and _entry_str(entry, "user_notes"):
                paper.user_notes = _entry_str(entry, "user_notes")
            if not paper.user_tags and _entry_list(entry, "user_tags"):
                paper.user_tags = _entry_list(entry, "user_tags")
            if not paper.openalex_id and _entry_str(entry, "openalex_id"):
                paper.openalex_id = _entry_str(entry, "openalex_id")
            if not paper.referenced_works and _entry_list(entry, "referenced_works"):
                paper.referenced_works = _entry_list(entry, "referenced_works")
            linked += 1

        if not PaperCollection.query.filter_by(paper_id=paper.id, collection_id=collection.id).first():
            db.session.add(PaperCollection(paper_id=paper.id, collection_id=collection.id))
        imported_papers.append(paper)

    # Annotations travel with their paper, but only onto papers that have none
    # locally yet (fill-only, like notes/tags). Invalid entries are skipped —
    # annotations are best-effort decoration, not worth failing the import.
    annotated_ids = set()
    for entry in manifest.get("annotations") or []:
        if not isinstance(entry, dict):
            continue
        index = entry.get("paper")
        page = entry.get("page")
        kind = entry.get("kind")
        rects = entry.get("rects")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(imported_papers)
            or not isinstance(page, int)
            or isinstance(page, bool)
            or page < 1
            or kind not in ("highlight", "comment")
            or not isinstance(rects, list)
            or not all(
                isinstance(r, dict) and all(isinstance(r.get(k), int | float) and 0 <= r[k] <= 1 for k in ("x", "y", "w", "h"))
                for r in rects
            )
        ):
            continue
        paper = imported_papers[index]
        if paper.id not in annotated_ids and PaperAnnotation.query.filter_by(paper_id=paper.id).first():
            continue
        annotated_ids.add(paper.id)
        db.session.add(
            PaperAnnotation(
                paper_id=paper.id,
                page=page,
                kind=kind,
                rects=rects,
                color=entry.get("color") if isinstance(entry.get("color"), str) else None,
                note=_entry_str(entry, "note"),
            )
        )

    db.session.commit()
    edges = sync_citation_edges()
    return collection, {"created": created, "linked": linked, "edges": edges}
