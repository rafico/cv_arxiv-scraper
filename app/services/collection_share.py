"""Portable collection bundles: export one collection (papers + notes + tags
+ reference ids) as plain JSON, and import such a bundle as a new collection.

No PDFs and no archive format on purpose — papers are re-fetchable from
arXiv, so a bundle is a single JSON document and the whole zip-hardening
surface disappears. Citation edges are not shipped either: each paper's
referenced_works travels with it, so sync_citation_edges() reproduces the
edges exactly on the receiving side.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)

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
    from app.models import Collection, Paper, PaperCollection, db

    collection = db.session.get(Collection, collection_id)
    if collection is None:
        raise ValueError("Collection not found")
    papers = (
        Paper.query.join(PaperCollection, PaperCollection.paper_id == Paper.id)
        .filter(PaperCollection.collection_id == collection.id)
        .order_by(Paper.id)
        .all()
    )
    return {
        "bundle_version": BUNDLE_VERSION,
        "collection": {
            "name": collection.name,
            "description": collection.description or "",
            "color": collection.color,
        },
        "papers": [{field: getattr(p, field) for field in _PAPER_FIELDS} for p in papers],
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
    if (
        not isinstance(collection, dict)
        or not isinstance(collection.get("name"), str)
        or not collection["name"].strip()
    ):
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


def arxiv_bundle_entry(
    arxiv_id: str, title: str, authors: list[str], abstract: str, publication_date: str, categories: list[str]
) -> dict:
    """A bundle paper entry for arXiv API metadata, so seeded ids (the import-ids
    route) and topic queries (``cv-arxiv-sync --query``) import like bundles."""
    from app.services.summary import generate_summary

    return {
        "arxiv_id": arxiv_id,
        "title": title or arxiv_id,
        "authors": ", ".join(authors),
        "link": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_link": f"https://arxiv.org/pdf/{arxiv_id}",
        "abstract_text": abstract,
        "summary_text": generate_summary(title, abstract),
        "publication_date": publication_date,
        "categories": categories,
    }


def import_collection(manifest: object, *, into=None, embed_max: int = MAX_BUNDLE_PAPERS):
    """Import a bundle as a new collection (or ``into`` an existing one);
    returns (collection, stats dict).

    Papers are global here (unlike per-dataset stores): an entry matching a
    local paper (by arxiv_id, then link) links into the collection and
    only fills user_notes/user_tags where locally empty — imports never
    overwrite local data. Unmatched entries become new Paper rows, embedded
    after the commit so search and "Suggest similar" can find them — unless
    there are more than ``embed_max`` (stats ``embedded`` is then False).
    """
    from datetime import date

    from flask import current_app

    from app.models import Collection, Paper, PaperCollection, db
    from app.services.citation_graph import sync_citation_edges

    manifest = _validate(manifest)

    collection = into
    if collection is None:
        collection = Collection(
            name=_unique_collection_name(manifest["collection"]["name"].strip()),
            description=str(manifest["collection"].get("description") or ""),
            color=manifest["collection"].get("color") if isinstance(manifest["collection"].get("color"), str) else None,
        )
        db.session.add(collection)
        db.session.flush()

    linked = 0
    new_ids: list[int] = []
    new_texts: list[str] = []
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
            new_ids.append(paper.id)
            new_texts.append(f"{paper.title} {paper.abstract_text or ''}")
        else:
            # Fill-only merge: never overwrite local data with imported data.
            if not paper.user_notes and _entry_str(entry, "user_notes"):
                paper.user_notes = _entry_str(entry, "user_notes")
            if not paper.user_tags and _entry_list(entry, "user_tags"):
                paper.user_tags = _entry_list(entry, "user_tags")
            if not paper.openalex_id and _entry_str(entry, "openalex_id"):
                paper.openalex_id = _entry_str(entry, "openalex_id")
            if not paper.semantic_scholar_id and _entry_str(entry, "semantic_scholar_id"):
                paper.semantic_scholar_id = _entry_str(entry, "semantic_scholar_id")
            if not paper.referenced_works and _entry_list(entry, "referenced_works"):
                paper.referenced_works = _entry_list(entry, "referenced_works")
            linked += 1

        if not PaperCollection.query.filter_by(paper_id=paper.id, collection_id=collection.id).first():
            db.session.add(PaperCollection(paper_id=paper.id, collection_id=collection.id))

    db.session.commit()
    edges = sync_citation_edges()
    embedded = len(new_ids) <= embed_max
    if new_ids and embedded:
        # Only this import's papers, via the scrape path's reload-append-save under the
        # index locks: saving the process singleton would write its stale matrix over
        # vectors a concurrent scrape or CLI just added.
        from app.services.embeddings import add_papers_to_index, get_embedding_service, reset_embedding_service
        from app.services.scrape_engine import _INDEX_WRITE_LOCK, _NATIVE_STAGE_TIMEOUT
        from app.services.subprocess_runner import run_isolated

        try:
            index_dir = str(get_embedding_service(current_app._get_current_object()).index_dir)
            with _INDEX_WRITE_LOCK:
                run_isolated(add_papers_to_index, index_dir, new_ids, new_texts, timeout=_NATIVE_STAGE_TIMEOUT)
                reset_embedding_service()
        except Exception:
            LOGGER.warning("Embedding imported papers failed (non-fatal)", exc_info=True)
    elif new_ids:
        LOGGER.info("Imported %d papers unembedded; run `cv-arxiv-backfill embeddings`", len(new_ids))
    return collection, {"created": len(new_ids), "linked": linked, "edges": edges, "embedded": embedded}
