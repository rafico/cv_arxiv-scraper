"""CLI entry point for the ArXiv CV scraper."""

from app import create_app
from app.models import Paper
from app.scraper import run_scrape


def _load_latest_matched(limit):
    return (
        Paper.query.order_by(Paper.scraped_date.desc(), Paper.id.desc())
        .limit(limit)
        .all()
    )


def _print_summary(result):
    print("\n===== Matched Articles =====")
    print(
        f"New: {result['new_papers']} | "
        f"Duplicates skipped: {result['duplicates_skipped']} | "
        f"Total matched: {result['total_matched']} / {result['total_in_feed']}"
    )
    print("=" * 50)


def _print_paper(index, paper):
    print(
        f"\n{index}. MATCHED PAPER\n"
        f"Match Type: {paper.match_type}\n"
        f"Title: {paper.title}\n"
        f"Authors: {paper.authors}\n"
        f"ArXiv Link: {paper.link}\n"
        f"PDF Link: {paper.pdf_link}\n"
        f"Publication Date: {paper.publication_date}\n"
        "Matched Terms:"
    )

    for term in paper.matched_terms.split(", "):
        print(f"  - {term}")

    print("-" * 50)


def main():
    app = create_app()
    result = run_scrape(app)

    with app.app_context():
        papers = _load_latest_matched(result["total_matched"])

    _print_summary(result)
    for idx, paper in enumerate(papers, start=1):
        _print_paper(idx, paper)


if __name__ == "__main__":
    main()
