"""CLI entry point for sending the daily email digest.

Usage:
    python email_digest.py              # scrape + send digest
    python email_digest.py --dry-run    # scrape + build email but don't send
    python email_digest.py --send-only  # send digest of already-scraped papers

Setup:
    1. pip install google-auth google-auth-oauthlib google-api-python-client
    2. Download credentials.json from Google Cloud Console
    3. python gmail_auth_setup.py       # one-time OAuth consent
    4. Set email.recipient in config.yaml

Cron example (daily at 08:00):
    0 8 * * * cd /path/to/cv_arxiv-scraper && /path/to/venv/bin/python email_digest.py
"""

import argparse
import sys

from app import create_app
from app.scraper import run_scrape
from app.services.email_digest import send_digest


def main() -> None:
    parser = argparse.ArgumentParser(description="ArXiv CV daily email digest")
    parser.add_argument("--dry-run", action="store_true", help="Build the email but do not actually send it")
    parser.add_argument("--send-only", action="store_true", help="Skip scraping — send digest from papers already in the database")
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
        sys.exit(1)

    status = "sent" if info["sent"] else "prepared (dry run)"
    print(f"Digest {status} — {info['papers_count']} papers → {info['recipient']}")


if __name__ == "__main__":
    main()
