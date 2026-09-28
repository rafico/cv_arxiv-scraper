"""CLI entry point for chunked historical sync runs."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Iterator
from datetime import date, datetime, time, timedelta

import requests

from app import create_app
from app.ingest.scrape_engine import execute_historical_scrape
from app.models import SyncState, db
from app.search_.text import now_utc
from app.services.collection_share import BUNDLE_VERSION, MAX_BUNDLE_PAPERS, arxiv_bundle_entry, import_collection
from app.services.enrichment_providers.semantic_scholar import search_arxiv_papers
from app.services.ingest import ArxivApiBackend
from app.services.ingest.arxiv_api_backend import ArxivRefused
from app.services.mcp_tools import _resolve_or_create_collection

CHUNK_DAYS = 7
# Mirrors the max_results cap execute_historical_scrape passes to the ingest
# orchestrator (BACKFILL mode). When a chunk's feed count reaches this cap the
# oldest papers were silently truncated, so we must warn instead of recording the
# chunk as fully synced.
INGEST_CHUNK_CAP = 2000
Summary = dict[str, int]


def parse_date_arg(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Expected YYYY-MM-DD.") from exc


def iter_date_chunks(start_dt: date, end_dt: date, *, chunk_days: int = CHUNK_DAYS) -> Iterator[tuple[date, date]]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")
    if end_dt < start_dt:
        raise ValueError("end_dt must be on or after start_dt")

    chunk_start = start_dt
    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end_dt)
        yield chunk_start, chunk_end
        chunk_start = chunk_end + timedelta(days=1)


def chunk_end_timestamp(end_dt: date) -> datetime:
    return datetime.combine(end_dt, time.max)


def upsert_sync_state(
    category: str,
    *,
    synced_through: date,
    paper_count: int,
    synced_at: datetime | None = None,
) -> SyncState:
    state = SyncState.query.filter_by(category=category).one_or_none()
    if state is None:
        state = SyncState(category=category)
        db.session.add(state)

    state.last_synced_submitted_at = chunk_end_timestamp(synced_through)
    state.last_synced_updated_at = synced_at or now_utc()
    state.last_synced_paper_count = paper_count
    state.last_cursor_page = None
    state.last_cursor_arxiv_id = None
    db.session.commit()
    return state


def _empty_summary() -> Summary:
    return {
        "new_papers": 0,
        "duplicates_skipped": 0,
        "total_matched": 0,
        "total_in_feed": 0,
    }


def run_sync(
    app,
    *,
    category: str,
    start_dt: date,
    end_dt: date,
    chunk_days: int = CHUNK_DAYS,
    emit: Callable[[str], None] = print,
) -> Summary:
    chunks = list(iter_date_chunks(start_dt, end_dt, chunk_days=chunk_days))
    aggregate = _empty_summary()
    with app.app_context():
        emit(
            f"Starting sync for {category}: {start_dt.isoformat()} -> {end_dt.isoformat()} "
            f"across {len(chunks)} chunk(s)"
        )

        for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            emit(f"[{index}/{len(chunks)}] Syncing {category} {chunk_start.isoformat()} -> {chunk_end.isoformat()}...")
            summary = execute_historical_scrape(app, [category], chunk_start, chunk_end)
            if int(summary.get("total_in_feed", 0)) >= INGEST_CHUNK_CAP:
                # The ingest layer hard-truncates at the cap (descending order, so the
                # OLDEST papers in this window were silently dropped). Recording the
                # chunk as synced — or letting later chunks overwrite synced_through —
                # would turn the truncation into a permanent invisible gap. Stop here
                # without advancing state so a re-run with a smaller --chunk-days
                # re-fetches the window.
                for key in aggregate:
                    aggregate[key] += int(summary.get(key, 0))
                emit(
                    f"[{index}/{len(chunks)}] ERROR: chunk hit the {INGEST_CHUNK_CAP}-paper ingest cap "
                    f"({chunk_start.isoformat()} -> {chunk_end.isoformat()}); the oldest papers in this "
                    "window were dropped. Sync state was NOT advanced — re-run this range with a smaller "
                    "--chunk-days."
                )
                break
            upsert_sync_state(
                category,
                synced_through=chunk_end,
                paper_count=summary["total_in_feed"],
            )

            for key in aggregate:
                aggregate[key] += int(summary.get(key, 0))

            emit(
                f"[{index}/{len(chunks)}] Done: "
                f"{summary['new_papers']} new, "
                f"{summary['duplicates_skipped']} duplicates, "
                f"{summary['total_matched']}/{summary['total_in_feed']} matched"
            )

        emit(
            "Sync complete: "
            f"{aggregate['new_papers']} new, "
            f"{aggregate['duplicates_skipped']} duplicates, "
            f"{aggregate['total_matched']}/{aggregate['total_in_feed']} matched"
        )
    return aggregate


def run_query_import(
    app,
    *,
    query: str,
    collection: str,
    start_dt: date,
    end_dt: date,
    max_results: int = 1000,
    categories: list[str] | None = None,
    emit: Callable[[str], None] = print,
) -> Summary:
    """Import an arXiv search straight into a collection (created on miss).

    Bypasses the scrape pipeline (no PDF downloads, no whitelist matching) and
    never calls run_sync/upsert_sync_state: a topic query says nothing about how
    far a category has been synced. When arXiv refuses the search, Semantic Scholar
    runs it instead.
    """
    with app.app_context():
        emit(f"Querying arXiv {start_dt.isoformat()} -> {end_dt.isoformat()}: {query}")
        try:
            candidates = ArxivApiBackend().fetch(
                categories=categories or [],
                start_dt=start_dt,
                end_dt=end_dt,
                max_results=max_results,
                query=query,
            )
        except ArxivRefused as exc:
            # S2 has no arXiv categories, only fieldsOfStudy: cs.* becomes all of Computer
            # Science (warned), anything else would be searched as the wrong field (refused).
            cats = [*(categories or []), *re.findall(r'\bcat:"?([\w.*-]+)', query)]
            outside_cs = [c for c in cats if c.split(".")[0] != "cs"]
            if outside_cs:
                raise ValueError(
                    f"arXiv refused the query (HTTP {exc.status}) and the Semantic Scholar fallback only "
                    f"covers Computer Science, not {', '.join(outside_cs)}; retry once arXiv accepts the query"
                ) from exc
            candidates = search_arxiv_papers(query, start_dt, end_dt, max_results)
            emit(f"arXiv refused the query (HTTP {exc.status}); used Semantic Scholar search instead")
            if cats:
                emit(
                    "WARNING: Semantic Scholar has no arXiv categories; --category/cat: filters were "
                    "replaced by all of Computer Science."
                )
            if re.search(r"\bau:", query):
                emit("WARNING: Semantic Scholar has no author search; au: names were matched as title/abstract words.")
        if len(candidates) >= max_results:
            emit(
                f"WARNING: hit --max-results {max_results}. Results come newest first, so the OLDEST "
                "matches were dropped. Narrow the query or --from/--to, or raise --max-results."
            )
        papers = [
            arxiv_bundle_entry(c.arxiv_id, c.title, c.authors_list, c.abstract, c.publication_date, c.categories)
            | {"semantic_scholar_id": c.semantic_scholar_id}
            for c in candidates
            if c.arxiv_id
        ]
        if not papers:
            emit("No matching papers; nothing imported.")
            return {"created": 0, "linked": 0, "edges": 0}
        target, _created = _resolve_or_create_collection(collection)
        if target is None:
            raise ValueError(f"Collection {collection} not found")
        manifest = {"bundle_version": BUNDLE_VERSION, "collection": {"name": target.name}, "papers": papers}
        _target, stats = import_collection(manifest, into=target)
        emit(f"Imported into '{target.name}': {stats['created']} new, {stats['linked']} already stored")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync arXiv papers over a historical date range, or import an arXiv search into a collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "topic query example (OR in synonyms: matching is lexical):\n"
            '  cv-arxiv-sync --query \'abs:"open-vocabulary segmentation" OR abs:"open-vocabulary semantic '
            "segmentation\"' --collection OVS --from 2019-01-01"
        ),
    )
    parser.add_argument("--from", dest="start_dt", type=parse_date_arg, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="end_dt", type=parse_date_arg, help="End date (YYYY-MM-DD)")
    parser.add_argument("--category", help="arXiv category, for example cs.CV")
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=CHUNK_DAYS,
        help=f"Chunk size in days, defaults to {CHUNK_DAYS}",
    )
    parser.add_argument(
        "--query",
        help="arXiv API search query (ti:, abs:, au:, AND/OR); imports matches into --collection "
        "instead of syncing. --from/--to default to all of arXiv, --category is optional",
    )
    parser.add_argument("--collection", help="Collection name (created if missing) or id, for --query")
    parser.add_argument(
        "--max-results", type=int, default=1000, help="Cap on --query results (newest first), defaults to 1000"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.query:
        if not args.collection:
            parser.error("--query requires --collection")
        if not 0 < args.max_results <= MAX_BUNDLE_PAPERS:
            parser.error(f"--max-results must be between 1 and {MAX_BUNDLE_PAPERS}")
    elif not (args.start_dt and args.end_dt and args.category):
        parser.error("--from, --to and --category are required (or use --query)")

    app = create_app()
    try:
        if args.query:
            run_query_import(
                app,
                query=args.query,
                collection=args.collection,
                start_dt=args.start_dt or date(1991, 1, 1),  # arXiv's first year
                end_dt=args.end_dt or now_utc().date(),
                max_results=args.max_results,
                categories=[args.category] if args.category else None,
            )
        else:
            run_sync(
                app,
                category=args.category,
                start_dt=args.start_dt,
                end_dt=args.end_dt,
                chunk_days=args.chunk_days,
            )
    except (RuntimeError, ValueError, OverflowError, requests.RequestException) as exc:
        # OverflowError: an oversized --chunk-days overflows date/timedelta arithmetic
        # in iter_date_chunks; report it cleanly instead of dumping a traceback.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
