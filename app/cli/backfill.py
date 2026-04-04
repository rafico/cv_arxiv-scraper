"""CLI entry point for selective enrichment backfills."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from app import create_app
from app.ingest.http_client import create_session
from app.models import Paper, db
from app.search_.text import now_utc

Emit = Callable[[str], None]

DEFAULT_BATCH_SIZE = 50
DEFAULT_DELAY_SECONDS = 1.0
EMBEDDINGS_BATCH_SIZE = 64


def _paper_index_paths(index_dir: Path) -> tuple[Path, Path]:
    return index_dir / "papers.index", index_dir / "id_map.json"


def _remove_paper_index_files(index_dir: Path) -> int:
    removed = 0
    for path in (
        index_dir / "papers.index",
        index_dir / "id_map.json",
        index_dir / "papers.index.tmp",
        index_dir / "id_map.json.tmp",
    ):
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def run_embeddings_backfill(app, *, batch_size: int = EMBEDDINGS_BATCH_SIZE, emit: Emit = print) -> int:
    from app.search_.embed_backfill import backfill_embeddings

    emit(f"Backfilling embeddings with batch size {batch_size}...")
    added = backfill_embeddings(app, batch_size=batch_size)
    emit(f"Embeddings backfill complete: {added} added")
    return added


def rebuild_semantic_index(app, *, batch_size: int = EMBEDDINGS_BATCH_SIZE, emit: Emit = print) -> int:
    from app.search_.embeddings import EmbeddingService, reset_embedding_service

    index_dir = Path(app.config["FAISS_INDEX_DIR"])
    index_dir.mkdir(parents=True, exist_ok=True)
    emit(f"Rebuilding semantic index with batch size {batch_size}...")

    batch_number = 0
    total_indexed = 0

    with tempfile.TemporaryDirectory(prefix="paper-index-rebuild-", dir=index_dir) as staging_dir:
        service = EmbeddingService(Path(staging_dir))

        with app.app_context():
            offset = 0
            while True:
                papers = Paper.query.order_by(Paper.id).offset(offset).limit(batch_size).all()
                if not papers:
                    break

                batch_number += 1
                paper_ids = [paper.id for paper in papers]
                texts = [f"{paper.title} {paper.abstract_text or ''}" for paper in papers]
                added = service.add_papers(paper_ids, texts)
                total_indexed += added
                emit(
                    f"Index rebuild batch {batch_number}: indexed {added}/{len(papers)} papers (total {total_indexed})"
                )
                offset += batch_size

        service.save()
        staging_index_path, staging_id_map_path = _paper_index_paths(Path(staging_dir))
        final_index_path, final_id_map_path = _paper_index_paths(index_dir)
        removed = _remove_paper_index_files(index_dir)

        os.replace(staging_index_path, final_index_path)
        os.replace(staging_id_map_path, final_id_map_path)

    reset_embedding_service()
    emit(
        f"Semantic index rebuild complete: {total_indexed} indexed "
        f"across {batch_number} batch(es); replaced {removed} existing file(s)"
    )
    return total_indexed


def backfill_citations(
    app,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    emit: Emit = print,
) -> int:
    from app.enrich.citations import fetch_citations_batch

    total_updated = 0
    last_seen_id = 0
    session = create_session(pool_size=1, scraper_config=app.config.get("SCRAPER_CONFIG"), rate_limit_profile="bulk")

    try:
        with app.app_context():
            while True:
                papers = (
                    Paper.query.filter(
                        Paper.id > last_seen_id,
                        Paper.arxiv_id.is_not(None),
                        Paper.citation_count.is_(None),
                    )
                    .order_by(Paper.id)
                    .limit(batch_size)
                    .all()
                )
                if not papers:
                    break

                last_seen_id = papers[-1].id
                arxiv_ids = [paper.arxiv_id for paper in papers if paper.arxiv_id]
                citation_data = fetch_citations_batch(arxiv_ids, session=session)
                updated_now = 0
                timestamp = now_utc()

                for paper in papers:
                    data = citation_data.get(paper.arxiv_id or "")
                    if not data:
                        continue

                    paper.citation_count = data.get("citation_count")
                    paper.influential_citation_count = data.get("influential_citation_count")
                    paper.semantic_scholar_id = data.get("semantic_scholar_id")
                    if paper.citation_count is not None:
                        paper.citation_source = "semantic_scholar"
                        paper.citation_provenance = {
                            "source": "semantic_scholar",
                            "updated_at": timestamp.isoformat(),
                        }
                        paper.citation_updated_at = timestamp
                    updated_now += 1

                db.session.commit()
                total_updated += updated_now
                emit(
                    f"Citations batch through paper {last_seen_id}: "
                    f"updated {updated_now}/{len(papers)} papers (total {total_updated})"
                )
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
    finally:
        session.close()

    return total_updated


def backfill_openalex(
    app,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    emit: Emit = print,
) -> int:
    from app.enrich.openalex import fetch_openalex_batch

    total_updated = 0
    last_seen_id = 0
    session = create_session(pool_size=1, scraper_config=app.config.get("SCRAPER_CONFIG"), rate_limit_profile="bulk")
    email = ((app.config.get("SCRAPER_CONFIG") or {}).get("openalex") or {}).get("email") or None

    try:
        with app.app_context():
            while True:
                papers = (
                    Paper.query.filter(
                        Paper.id > last_seen_id,
                        Paper.arxiv_id.is_not(None),
                        Paper.openalex_id.is_(None),
                    )
                    .order_by(Paper.id)
                    .limit(batch_size)
                    .all()
                )
                if not papers:
                    break

                last_seen_id = papers[-1].id
                arxiv_ids = [paper.arxiv_id for paper in papers if paper.arxiv_id]
                openalex_data = fetch_openalex_batch(arxiv_ids, session=session, email=email)
                updated_now = 0
                timestamp = now_utc()

                for paper in papers:
                    data = openalex_data.get(paper.arxiv_id or "")
                    if not data:
                        continue

                    paper.openalex_id = data.get("openalex_id")
                    paper.openalex_topics = data.get("openalex_topics", [])
                    paper.oa_status = data.get("oa_status")
                    paper.openalex_cited_by_count = data.get("openalex_cited_by_count")
                    paper.referenced_works_count = data.get("referenced_works_count")
                    if paper.citation_count is None and paper.openalex_cited_by_count is not None:
                        paper.citation_source = "openalex"
                        paper.citation_provenance = {
                            "source": "openalex",
                            "updated_at": timestamp.isoformat(),
                        }
                    updated_now += 1

                db.session.commit()
                total_updated += updated_now
                emit(
                    f"OpenAlex batch through paper {last_seen_id}: "
                    f"updated {updated_now}/{len(papers)} papers (total {total_updated})"
                )
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
    finally:
        session.close()

    return total_updated


def backfill_thumbnails(
    app,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    emit: Emit = print,
) -> int:
    from app.search_.thumbnail_generator import generate_thumbnail

    total_generated = 0
    last_seen_id = 0
    session = create_session(pool_size=1, scraper_config=app.config.get("SCRAPER_CONFIG"), rate_limit_profile="bulk")

    try:
        with app.app_context():
            static_dir = Path(app.static_folder)
            thumbnails_dir = static_dir / "thumbnails"

            while True:
                papers = (
                    Paper.query.filter(
                        Paper.id > last_seen_id,
                        Paper.arxiv_id.is_not(None),
                        Paper.pdf_link.is_not(None),
                    )
                    .order_by(Paper.id)
                    .limit(batch_size)
                    .all()
                )
                if not papers:
                    break

                last_seen_id = papers[-1].id
                generated_now = 0
                for paper in papers:
                    thumbnail_path = thumbnails_dir / f"{paper.arxiv_id}.png"
                    if thumbnail_path.exists():
                        continue

                    if generate_thumbnail(paper.arxiv_id, paper.pdf_link, static_dir, session=session):
                        generated_now += 1
                        total_generated += 1

                    if delay_seconds > 0:
                        time.sleep(delay_seconds)

                emit(
                    f"Thumbnail batch through paper {last_seen_id}: "
                    f"generated {generated_now}/{len(papers)} thumbnails (total {total_generated})"
                )
    finally:
        session.close()

    return total_generated


def run_all_backfills(
    app,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    emit: Emit = print,
) -> dict[str, int]:
    results = {
        "embeddings": run_embeddings_backfill(app, batch_size=EMBEDDINGS_BATCH_SIZE, emit=emit),
        "citations": backfill_citations(app, batch_size=batch_size, delay_seconds=delay_seconds, emit=emit),
        "openalex": backfill_openalex(app, batch_size=batch_size, delay_seconds=delay_seconds, emit=emit),
        "thumbnails": backfill_thumbnails(app, batch_size=batch_size, delay_seconds=delay_seconds, emit=emit),
    }
    emit(
        "All backfills complete: "
        f"embeddings={results['embeddings']}, "
        f"citations={results['citations']}, "
        f"openalex={results['openalex']}, "
        f"thumbnails={results['thumbnails']}"
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run selective enrichment backfills")
    subparsers = parser.add_subparsers(dest="command", required=True)

    embeddings = subparsers.add_parser("embeddings", help="Backfill missing embeddings")
    embeddings.add_argument("--batch-size", type=int, default=EMBEDDINGS_BATCH_SIZE)
    index_rebuild = subparsers.add_parser("index-rebuild", help="Rebuild the semantic paper index from the DB")
    index_rebuild.add_argument("--batch-size", type=int, default=EMBEDDINGS_BATCH_SIZE)

    for command in ("citations", "openalex", "thumbnails", "all"):
        subparser = subparsers.add_parser(command, help=f"Run {command} backfill")
        subparser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
        subparser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    app = create_app()

    try:
        if args.command == "embeddings":
            run_embeddings_backfill(app, batch_size=args.batch_size)
        elif args.command == "index-rebuild":
            rebuild_semantic_index(app, batch_size=args.batch_size)
        elif args.command == "citations":
            backfill_citations(app, batch_size=args.batch_size, delay_seconds=args.delay)
        elif args.command == "openalex":
            backfill_openalex(app, batch_size=args.batch_size, delay_seconds=args.delay)
        elif args.command == "thumbnails":
            backfill_thumbnails(app, batch_size=args.batch_size, delay_seconds=args.delay)
        else:
            run_all_backfills(app, batch_size=args.batch_size, delay_seconds=args.delay)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
