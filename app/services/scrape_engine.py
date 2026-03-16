"""Core scrape execution pipeline."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Callable

from app.services.enrichment import (
    enrich_entries_with_api_metadata,
    extract_affiliation_text,
    now_utc,
    parse_feed_entries,
)
from app.services.http_client import request_with_backoff
from app.services.matching import (
    MATCH_PRIORITY,
    check_author_match,
    check_whitelist_match,
    dedupe_preserve_order,
)
from app.constants import DEFAULT_MAX_WORKERS
from app.services.ranking import compute_paper_score
from app.services.summary import extract_topic_tags, generate_summary

LOGGER = logging.getLogger(__name__)


EventCallback = Callable[[str, dict], None] | None


def _emit(callback: EventCallback, event: str, data: dict) -> None:
    if callback:
        callback(event, data)


def _build_result(
    entry_data: dict,
    category_matches: dict[str, list[str]],
) -> dict:
    """Assemble a result dict from an entry and its matches."""
    match_types = [name for name, terms in category_matches.items() if terms]
    matched_terms = dedupe_preserve_order(
        term for terms in category_matches.values() for term in terms
    )
    title = entry_data["title"]
    abstract = entry_data.get("abstract", "")

    return {
        "arxiv_id": entry_data.get("arxiv_id"),
        "title": title,
        "authors": entry_data["author"],
        "link": entry_data["link"],
        "pdf_link": entry_data["link"].replace("/abs/", "/pdf/"),
        "abstract_text": abstract,
        "summary_text": generate_summary(title, abstract),
        "topic_tags": extract_topic_tags(title, abstract),
        "categories": entry_data.get("categories", []),
        "resource_links": entry_data.get("resource_links", []),
        "matches": matched_terms,
        "match_types": match_types,
        "match_type": " + ".join(match_types),
        "match_priority": min(MATCH_PRIORITY[name] for name in match_types),
        "paper_score": compute_paper_score(
            match_types=match_types,
            matched_terms_count=len(matched_terms),
            publication_dt=entry_data.get("publication_dt"),
            resource_count=len(entry_data.get("resource_links", [])),
        ),
        "publication_dt": entry_data.get("publication_dt"),
        "publication_date": entry_data.get("publication_date", "Date Unknown"),
    }


def _check_fast_matches(entry_data: dict, whitelists: dict) -> dict[str, list[str]]:
    """Check title and author matches — no network needed."""
    return {
        "Author": check_author_match(entry_data["authors_list"], whitelists["authors"]),
        "Title": check_whitelist_match(
            [entry_data["title"], entry_data.get("abstract", "")],
            whitelists["titles"],
        ),
    }


def _process_paper_entry(entry_data: dict, whitelists: dict, scraper_config: dict) -> dict | None:
    # Phase 1: fast check — title and author (no PDF download).
    fast_matches = _check_fast_matches(entry_data, whitelists)

    # Phase 2: download PDF and check affiliations for all papers.
    link = entry_data["link"]
    pdf_url = link.replace("/abs/", "/pdf/")
    affiliation_matches: list[str] = []
    try:
        pdf_response = request_with_backoff(
            "GET",
            pdf_url,
            timeout=30,
            attempts=scraper_config.get("pdf_attempts", 2),
            base_delay=1.0,
        )
        affiliation_text = extract_affiliation_text(
            pdf_response.content,
            lines_start=scraper_config.get("pdf_lines_start", 2),
            max_header_lines=scraper_config.get(
                "pdf_max_header_lines", scraper_config.get("pdf_lines_end", 50)
            ),
            smart_header=scraper_config.get("pdf_smart_header", True),
        )
        api_affiliations = entry_data.get("api_affiliations", "")
        affiliation_sources = [text for text in [affiliation_text, api_affiliations] if text]
        affiliation_matches = check_whitelist_match(affiliation_sources, whitelists["affiliations"])
    except Exception as exc:
        LOGGER.warning("Error fetching PDF for %s: %s", link, exc)

    category_matches = {**fast_matches, "Affiliation": affiliation_matches}

    if not any(category_matches.values()):
        return None

    return _build_result(entry_data, category_matches)


def _process_entries_parallel(entries: list[dict], whitelists: dict, scraper_config: dict):
    max_workers = max(1, int(scraper_config.get("max_workers", DEFAULT_MAX_WORKERS)))
    processed = 0
    matched = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_paper_entry, entry, whitelists, scraper_config): entry
            for entry in entries
        }

        for future in as_completed(futures):
            entry = futures[future]
            processed += 1
            result = None
            try:
                result = future.result()
            except Exception:
                LOGGER.exception(
                    "Unhandled worker exception while processing paper: %s (%s)",
                    entry.get("title"),
                    entry.get("link"),
                )

            if result:
                matched += 1

            yield processed, matched, result


def _sort_results(results: list[dict]) -> None:
    results.sort(
        key=lambda item: (
            float(item.get("paper_score", 0.0)),
            item.get("publication_dt") or date.min,
        ),
        reverse=True,
    )


def _save_results(app, results: list[dict]) -> tuple[int, int]:
    from app.models import Paper, db

    now = now_utc()
    today_str = now.date().isoformat()
    skipped = 0
    new_papers = []

    with app.app_context():
        # Check DB for links that slipped through pre-filter (race / direct call).
        links = [r["link"] for r in results]
        existing_links: set[str] = set()
        if links:
            existing_rows = db.session.query(Paper.link).filter(Paper.link.in_(links)).all()
            existing_links = {link for (link,) in existing_rows}

        seen: set[str] = set()
        for result in results:
            link = result["link"]
            if link in existing_links or link in seen:
                skipped += 1
                continue
            seen.add(link)

            new_papers.append(
                Paper(
                    arxiv_id=result.get("arxiv_id"),
                    title=result["title"],
                    authors=result["authors"],
                    link=result["link"],
                    pdf_link=result["pdf_link"],
                    abstract_text=result.get("abstract_text", ""),
                    summary_text=result.get("summary_text", ""),
                    topic_tags=result.get("topic_tags", []),
                    categories=result.get("categories", []),
                    resource_links=result.get("resource_links", []),
                    match_type=result["match_type"],
                    matched_terms=result["matches"],
                    paper_score=float(result.get("paper_score", 0.0)),
                    publication_date=result["publication_date"],
                    publication_dt=result.get("publication_dt"),
                    scraped_date=today_str,
                    scraped_at=now,
                )
            )

        if new_papers:
            db.session.add_all(new_papers)
            db.session.commit()

    return len(new_papers), skipped


def _build_summary(new_count: int, skipped: int, total_matched: int, total_in_feed: int) -> dict:
    return {
        "new_papers": new_count,
        "duplicates_skipped": skipped,
        "total_matched": total_matched,
        "total_in_feed": total_in_feed,
    }


def _get_existing_links(app, entries: list[dict]) -> set[str]:
    """Pre-check which links already exist in the DB to skip them early."""
    from app.models import Paper, db

    links = [entry["link"] for entry in entries]
    if not links:
        return set()

    with app.app_context():
        rows = db.session.query(Paper.link).filter(Paper.link.in_(links)).all()
        return {link for (link,) in rows}


def execute_scrape(app, event_callback: EventCallback = None) -> dict:
    config = app.config["SCRAPER_CONFIG"]
    whitelists = config["whitelists"]
    scraper_config = config["scraper"]

    _emit(event_callback, "status", {"phase": "feed", "message": "Fetching RSS feed..."})
    entries = parse_feed_entries(scraper_config["feed_url"])
    total_entries = len(entries)
    _emit(event_callback, "feed", {"total": total_entries})

    # Skip entries already in the database before doing any heavy work.
    existing_links = _get_existing_links(app, entries)
    pre_filtered = len(existing_links)
    if existing_links:
        entries = [e for e in entries if e["link"] not in existing_links]
        LOGGER.info("Skipped %d already-stored papers, %d new to process", pre_filtered, len(entries))

    _emit(
        event_callback,
        "status",
        {"phase": "affiliations", "message": "Fetching metadata from arXiv API..."},
    )
    enrich_entries_with_api_metadata(entries)

    _emit(
        event_callback,
        "status",
        {"phase": "processing", "message": f"Processing {len(entries)} papers..."},
    )

    results: list[dict] = []
    for processed, matched, result in _process_entries_parallel(entries, whitelists, scraper_config):
        payload = {"processed": processed, "total": total_entries, "matched": matched}
        if result:
            results.append(result)
            _emit(
                event_callback,
                "match",
                {
                    **payload,
                    "paper": {
                        "title": result["title"],
                        "match_type": result["match_type"],
                        "match_types": result["match_types"],
                        "matched_terms": result["matches"],
                    },
                },
            )
        else:
            _emit(event_callback, "progress", payload)

    _emit(event_callback, "status", {"phase": "saving", "message": "Saving to database..."})
    _sort_results(results)
    new_count, skipped = _save_results(app, results)
    summary = _build_summary(new_count, skipped + pre_filtered, len(results), total_entries)
    _emit(event_callback, "done", summary)

    LOGGER.info(
        "Scrape complete: %s new, %s duplicates, %s matched out of %s entries",
        new_count,
        skipped,
        len(results),
        total_entries,
    )
    return summary


def run_scrape(app) -> dict:
    return execute_scrape(app, event_callback=None)


def stream_or_start_scrape(app):
    """Compatibility wrapper implemented in job manager module."""
    from app.services.jobs import SCRAPE_JOB_MANAGER

    return SCRAPE_JOB_MANAGER.stream_for_request(app)
