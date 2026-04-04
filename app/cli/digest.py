"""CLI entry point for sending the daily email digest."""

from __future__ import annotations

import argparse
import sys

from app import create_app
from app.ingest.scrape_engine import run_scrape
from app.web.email_digest import send_digest


def main() -> None:
    parser = argparse.ArgumentParser(description="ArXiv CV daily email digest")
    parser.add_argument("--dry-run", action="store_true", help="Build the email but do not actually send it")
    parser.add_argument(
        "--send-only", action="store_true", help="Skip scraping; send digest from papers already in the database"
    )
    args = parser.parse_args()

    app = create_app()

    if not args.send_only:
        print("Running scrape...")
        result = run_scrape(app)
        print(
            f"Scrape complete: {result['new_papers']} new, "
            f"{result['duplicates_skipped']} duplicates, "
            f"{result['total_matched']}/{result['total_in_feed']} matched"
        )

    print("Preparing email digest...")
    try:
        info = send_digest(app, dry_run=args.dry_run)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    status = "sent" if info["sent"] else "prepared (dry run)"
    print(f"Digest {status} - {info['papers_count']} papers -> {info['recipient']}")


if __name__ == "__main__":
    main()
