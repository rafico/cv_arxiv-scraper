"""CLI entry point for exporting paper reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from app import create_app
from app.services.export import generate_html_report
from app.services.text import now_utc


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ArXiv CV papers to HTML")
    parser.add_argument("--timeframe", choices=("daily", "weekly", "monthly", "all"), default="daily", help="Dashboard timeframe to export")
    parser.add_argument("--output", help="Optional output file path")
    args = parser.parse_args()

    app = create_app()
    default_name = f"arxiv_report_{args.timeframe}_{now_utc().date().isoformat()}.html"
    output_path = Path(args.output) if args.output else Path(default_name)

    generate_html_report(app, timeframe=args.timeframe, output_path=output_path)
    print(f"Exported report to {output_path}")


if __name__ == "__main__":
    main()
