"""CLI entry point for chunked historical sync runs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator
from datetime import date, datetime, time, timedelta

from app import create_app
from app.models import SyncState, db
from app.services.scrape_engine import execute_historical_scrape
from app.services.text import now_utc

CHUNK_DAYS = 7
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync arXiv papers over a historical date range")
    parser.add_argument("--from", dest="start_dt", required=True, type=parse_date_arg, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="end_dt", required=True, type=parse_date_arg, help="End date (YYYY-MM-DD)")
    parser.add_argument("--category", required=True, help="arXiv category, for example cs.CV")
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=CHUNK_DAYS,
        help=f"Chunk size in days, defaults to {CHUNK_DAYS}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    app = create_app()
    try:
        run_sync(
            app,
            category=args.category,
            start_dt=args.start_dt,
            end_dt=args.end_dt,
            chunk_days=args.chunk_days,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
