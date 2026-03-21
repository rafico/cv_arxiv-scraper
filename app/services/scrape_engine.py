"""Core scrape execution pipeline."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import requests
from sqlalchemy.exc import IntegrityError

from app.services.enrichment import (
    fetch_recent_papers,
    enrich_entries_with_api_metadata,
    extract_affiliation_text,
    now_utc,
    parse_feed_entries,
)
from app.services.http_client import create_session, request_with_backoff
from app.services.llm_client import LLMClient, resolve_api_key
from app.services.matching import (
    MATCH_PRIORITY,
    check_author_match,
    check_whitelist_match,
    dedupe_preserve_order,
)
from app.constants import DEFAULT_MAX_WORKERS
from app.services.preferences import get_preferences
from app.services.ranking import compute_paper_score
from app.services.summary import extract_topic_tags, generate_llm_summary, generate_summary

LOGGER = logging.getLogger(__name__)


EventCallback = Callable[[str, dict], None] | None


def _emit(callback: EventCallback, event: str, data: dict) -> None:
    if callback:
        callback(event, data)


def _build_result(
    entry_data: dict,
    category_matches: dict[str, list[str]],
    llm_client: LLMClient | None = None,
    interests_text: str = "",
    config: dict | None = None,
) -> dict:
    """Assemble a result dict from an entry and its matches."""
    match_types = [name for name, terms in category_matches.items() if terms]
    matched_terms = dedupe_preserve_order(
        term for terms in category_matches.values() for term in terms
    )
    title = entry_data["title"]
    abstract = entry_data.get("abstract", "")
    summary_text = (
        generate_llm_summary(llm_client, title, abstract)
        if llm_client is not None
        else generate_summary(title, abstract)
    )
    llm_relevance_score = (
        llm_client.rate_relevance(title, abstract, interests_text)
        if llm_client is not None
        else None
    )

    topic_tags = extract_topic_tags(title, abstract)

    return {
        "arxiv_id": entry_data.get("arxiv_id"),
        "title": title,
        "authors": entry_data["author"],
        "link": entry_data["link"],
        "pdf_link": entry_data["link"].replace("/abs/", "/pdf/"),
        "abstract_text": abstract,
        "summary_text": summary_text,
        "topic_tags": topic_tags,
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
            llm_relevance_score=llm_relevance_score,
            config=config,
        ),
        "llm_relevance_score": llm_relevance_score,
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


def _process_paper_entry(
    entry_data: dict,
    whitelists: dict,
    scraper_config: dict,
    session: requests.Session | None = None,
    llm_client: LLMClient | None = None,
    interests_text: str = "",
    product_config: dict | None = None,
) -> dict | None:
    # Phase 1: fast check — title and author (no PDF download).
    fast_matches = _check_fast_matches(entry_data, whitelists)

    # Phase 2: download PDF and check affiliations.
    link = entry_data["link"]
    pdf_url = link.replace("/abs/", "/pdf/")
    affiliation_matches: list[str] = []
    
    api_affiliations = entry_data.get("api_affiliations", "")
    if api_affiliations:
        affiliation_matches = check_whitelist_match([api_affiliations], whitelists["affiliations"])

    if not affiliation_matches:
        try:
            pdf_response = request_with_backoff(
                "GET",
                pdf_url,
                timeout=30,
                attempts=scraper_config.get("pdf_attempts", 2),
                base_delay=1.0,
                session=session,
            )
            affiliation_text = extract_affiliation_text(
                pdf_response.content,
                lines_start=scraper_config.get("pdf_lines_start", 2),
                max_header_lines=scraper_config.get(
                    "pdf_max_header_lines", scraper_config.get("pdf_lines_end", 50)
                ),
                smart_header=scraper_config.get("pdf_smart_header", True),
            )
            if affiliation_text:
                affiliation_matches = check_whitelist_match([affiliation_text], whitelists["affiliations"])
        except Exception as exc:
            LOGGER.warning("Error fetching PDF for %s: %s", link, exc)

    category_matches = {**fast_matches, "Affiliation": affiliation_matches}

    if not any(category_matches.values()):
        return None

    # Check mute filters before expensive LLM calls.
    preferences = get_preferences(product_config)
    muted = preferences["muted"]
    if check_author_match(entry_data["authors_list"], muted["authors"]):
        return None
    if check_whitelist_match([entry_data.get("api_affiliations", "")], muted["affiliations"]):
        return None
    topic_tags = extract_topic_tags(entry_data["title"], entry_data.get("abstract", ""))
    if check_whitelist_match(topic_tags, muted["topics"]):
        return None

    return _build_result(
        entry_data,
        category_matches,
        llm_client=llm_client,
        interests_text=interests_text,
        config=product_config,
    )


def _process_entries_parallel(
    entries: list[dict],
    whitelists: dict,
    scraper_config: dict,
    session: requests.Session,
    llm_client: LLMClient | None = None,
    interests_text: str = "",
    product_config: dict | None = None,
):
    max_workers = max(1, int(scraper_config.get("max_workers", DEFAULT_MAX_WORKERS)))
    processed = 0
    matched = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_paper_entry,
                entry,
                whitelists,
                scraper_config,
                session,
                llm_client,
                interests_text,
                product_config,
            ): entry
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


def _identity_keys(data: dict) -> set[str]:
    keys = set()
    link = data.get("link")
    arxiv_id = data.get("arxiv_id")
    if isinstance(link, str) and link:
        keys.add(link)
    if isinstance(arxiv_id, str) and arxiv_id:
        keys.add(arxiv_id)
    return keys


def _save_results(app, results: list[dict]) -> tuple[int, int]:
    from app.models import Paper, db
    from app.services.related import find_duplicates

    now = now_utc()
    today_str = now.date().isoformat()
    skipped = 0
    new_count = 0

    with app.app_context():
        existing_keys = _get_existing_ids(app, results)

        # Build title map for duplicate detection.
        existing_titles: dict[int, str] = {}
        for pid, ptitle in db.session.query(Paper.id, Paper.title).yield_per(500):
            existing_titles[pid] = ptitle

        seen_keys: set[str] = set()
        for result in results:
            identity_keys = _identity_keys(result)
            if any(key in existing_keys or key in seen_keys for key in identity_keys):
                skipped += 1
                continue

            # Check for near-duplicate titles.
            duplicate_of_id = None
            dups = find_duplicates(result["title"], existing_titles)
            if dups:
                duplicate_of_id = dups[0][0]

            seen_keys.update(identity_keys)
            paper = Paper(
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
                llm_relevance_score=result.get("llm_relevance_score"),
                publication_date=result["publication_date"],
                publication_dt=result.get("publication_dt"),
                scraped_date=today_str,
                scraped_at=now,
                duplicate_of_id=duplicate_of_id,
                citation_count=result.get("citation_count"),
                influential_citation_count=result.get("influential_citation_count"),
                semantic_scholar_id=result.get("semantic_scholar_id"),
                citation_updated_at=result.get("citation_updated_at"),
            )
            db.session.add(paper)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                skipped += 1
                continue

            existing_keys.update(identity_keys)
            new_count += 1

    return new_count, skipped


def _generate_thumbnails(app, results: list[dict], session: requests.Session) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from app.services.thumbnail_generator import generate_thumbnail

    static_folder = app.static_folder if app.static_folder else Path(__file__).parent.parent / "static"

    def worker(res):
        arxiv_id = res.get("arxiv_id") or (res.get("link") or "").split("/")[-1]
        pdf_link = res.get("pdf_link")
        if arxiv_id and pdf_link:
            generate_thumbnail(arxiv_id, pdf_link, static_folder, session=session)

    with ThreadPoolExecutor(max_workers=3) as executor:
        for _ in executor.map(worker, results):
            pass


def _build_summary(new_count: int, skipped: int, total_matched: int, total_in_feed: int) -> dict:
    return {
        "new_papers": new_count,
        "duplicates_skipped": skipped,
        "total_matched": total_matched,
        "total_in_feed": total_in_feed,
    }


def _get_existing_ids(app, entries: list[dict]) -> set[str]:
    """Pre-check which links or arXiv ids already exist to skip them early."""
    from app.models import Paper, db

    links = [entry["link"] for entry in entries if entry.get("link")]
    arxiv_ids = [entry["arxiv_id"] for entry in entries if entry.get("arxiv_id")]
    if not links and not arxiv_ids:
        return set()

    filters = []
    if links:
        filters.append(Paper.link.in_(links))
    if arxiv_ids:
        filters.append(Paper.arxiv_id.in_(arxiv_ids))

    with app.app_context():
        rows = db.session.query(Paper.link, Paper.arxiv_id).filter(db.or_(*filters)).all()
        existing: set[str] = set()
        for link, arxiv_id in rows:
            if link:
                existing.add(link)
            if arxiv_id:
                existing.add(arxiv_id)
        return existing


def _build_llm_interests(whitelists: dict) -> str:
    parts = []
    if whitelists.get("titles"):
        parts.append(f"Title keywords: {', '.join(whitelists['titles'])}")
    if whitelists.get("authors"):
        parts.append(f"Authors: {', '.join(whitelists['authors'])}")
    if whitelists.get("affiliations"):
        parts.append(f"Affiliations: {', '.join(whitelists['affiliations'])}")
    return "; ".join(parts)


def _has_successful_scrape_today(app, today_start, tomorrow_start) -> bool:
    from app.models import ScrapeRun, db

    with app.app_context():
        return (
            db.session.query(ScrapeRun.id)
            .filter(
                ScrapeRun.status == "success",
                ScrapeRun.started_at >= today_start,
                ScrapeRun.started_at < tomorrow_start,
            )
            .first()
            is not None
        )


def _create_scrape_run(app, started_at, *, force: bool):
    from app.models import ScrapeRun, db

    with app.app_context():
        scrape_run = ScrapeRun(status="running", forced=force, started_at=started_at)
        db.session.add(scrape_run)
        db.session.commit()
        return scrape_run.id


def _finish_scrape_run(app, scrape_run_id: int | None, *, status: str) -> None:
    if scrape_run_id is None:
        return

    from app.models import ScrapeRun, db

    with app.app_context():
        scrape_run = db.session.get(ScrapeRun, scrape_run_id)
        if scrape_run is None:
            return
        scrape_run.status = status
        scrape_run.finished_at = now_utc()
        db.session.commit()


def _create_llm_client(app) -> tuple[LLMClient | None, str]:
    llm_config = app.config["SCRAPER_CONFIG"].get("llm", {})
    if not llm_config.get("enabled"):
        return None, ""

    provider = llm_config.get("provider", "openrouter")

    if provider == "ollama":
        api_key = "ollama"
        default_base_url = "http://localhost:11434/v1"
        default_model = "llama3"
    else:
        api_key = resolve_api_key(Path(app.config["LLM_KEY_PATH"]))
        if not api_key:
            LOGGER.warning("LLM is enabled but no API key is available")
            return None, ""
        default_base_url = "https://openrouter.ai/api/v1"
        default_model = "anthropic/claude-sonnet-4"

    try:
        client = LLMClient(
            api_key=api_key,
            model=llm_config.get("model", default_model),
            base_url=llm_config.get("base_url", default_base_url),
            max_concurrent=int(llm_config.get("max_concurrent", 4)),
        )
    except Exception as exc:
        LOGGER.warning("Unable to initialize LLM client: %s", exc)
        return None, ""

    interests_text = _build_llm_interests(app.config["SCRAPER_CONFIG"]["whitelists"])
    return client, interests_text


def _enrich_results_with_citations(
    results: list[dict],
    session: requests.Session,
    config: dict,
    now=None,
) -> None:
    """Enrich matched results with Semantic Scholar citation data (in-place)."""
    if not results:
        return

    from app.services.citations import fetch_citations_batch
    from app.services.ranking import compute_paper_score

    arxiv_ids = [res["arxiv_id"] for res in results if res.get("arxiv_id")]
    if not arxiv_ids:
        return

    if now is None:
        now = now_utc()

    citation_data = fetch_citations_batch(arxiv_ids, session=session)
    for res in results:
        arxiv_id = res.get("arxiv_id")
        if arxiv_id and arxiv_id in citation_data:
            data = citation_data[arxiv_id]
            res["citation_count"] = data.get("citation_count")
            res["influential_citation_count"] = data.get("influential_citation_count")
            res["semantic_scholar_id"] = data.get("semantic_scholar_id")
            if res["citation_count"] is not None:
                res["citation_updated_at"] = now
            res["paper_score"] = compute_paper_score(
                match_types=res.get("match_types", []),
                matched_terms_count=len(res.get("matches", [])),
                publication_dt=res.get("publication_dt"),
                resource_count=len(res.get("resource_links", [])),
                llm_relevance_score=res.get("llm_relevance_score"),
                citation_count=res.get("citation_count"),
                config=config,
            )


def execute_scrape(app, event_callback: EventCallback = None, force: bool = False) -> dict:
    config = app.config["SCRAPER_CONFIG"]
    whitelists = config["whitelists"]
    scraper_config = config["scraper"]
    now = now_utc()
    scrape_run_id: int | None = None

    if not force:
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow_start = today_start + timedelta(days=1)
        if _has_successful_scrape_today(app, today_start, tomorrow_start):
            payload = {
                **_build_summary(0, 0, 0, 0),
                "skipped": True,
                "reason": "Already scraped today",
            }
            _emit(event_callback, "skipped", payload)
            return payload

    scrape_run_id = _create_scrape_run(app, now, force=force)

    try:
        max_workers = max(1, int(scraper_config.get("max_workers", DEFAULT_MAX_WORKERS)))
        session = create_session(pool_size=max_workers)

        _emit(event_callback, "status", {"phase": "feed", "message": "Fetching RSS feed..."})

        # Collect feed URLs: primary config feed + any enabled FeedSource entries.
        feed_urls = scraper_config.get("feed_urls") or []
        if scraper_config.get("feed_url") and scraper_config["feed_url"] not in feed_urls:
            feed_urls.append(scraper_config["feed_url"])
        try:
            from app.models import FeedSource, db as _db

            with app.app_context():
                extra_sources = FeedSource.query.filter_by(enabled=True).all()
                for src in extra_sources:
                    if src.url not in feed_urls:
                        feed_urls.append(src.url)
        except Exception:
            pass  # FeedSource table may not exist yet.

        entries: list[dict] = []
        feed_errors: list[Exception] = []
        for feed_url in feed_urls:
            try:
                entries.extend(parse_feed_entries(feed_url, session=session))
            except Exception as exc:
                LOGGER.warning("Failed to parse feed %s: %s", feed_url, exc)
                feed_errors.append(exc)

        # If every feed failed, propagate the first error.
        if feed_errors and not entries and len(feed_errors) == len(feed_urls):
            raise feed_errors[0]
        rolling_window_days = max(0, int(scraper_config.get("rolling_window_days", 0)))
        if rolling_window_days > 0:
            _emit(
                event_callback,
                "status",
                {
                    "phase": "rolling_window",
                    "message": f"Loading papers from the past {rolling_window_days} days...",
                },
            )
            try:
                recent_entries = []
                for f_url in feed_urls:
                    recent = fetch_recent_papers(rolling_window_days, f_url, session=session)
                    recent_entries.extend(recent)
            except Exception as exc:
                LOGGER.warning("Rolling-window fetch failed: %s", exc)
                recent_entries = []

            merged_entries: dict[str, dict] = {}
            for entry in entries:
                merged_entries[entry.get("arxiv_id") or entry["link"]] = entry
            for entry in recent_entries:
                merged_entries.setdefault(entry.get("arxiv_id") or entry["link"], entry)
            entries = list(merged_entries.values())

        total_entries = len(entries)
        _emit(event_callback, "feed", {"total": total_entries})

        # Skip entries already in the database before doing any heavy work.
        existing_ids = _get_existing_ids(app, entries)
        pre_filtered = 0
        if existing_ids:
            filtered_entries = [
                entry
                for entry in entries
                if not _identity_keys(entry).intersection(existing_ids)
            ]
            pre_filtered = len(entries) - len(filtered_entries)
            entries = filtered_entries
            LOGGER.info(
                "Skipped %d already-stored papers, %d new to process",
                pre_filtered,
                len(entries),
            )

        llm_client, interests_text = _create_llm_client(app)

        _emit(
            event_callback,
            "status",
            {"phase": "affiliations", "message": "Fetching metadata from arXiv API..."},
        )
        enrich_entries_with_api_metadata(entries, session=session)

        _emit(
            event_callback,
            "status",
            {"phase": "processing", "message": f"Processing {len(entries)} papers..."},
        )

        results: list[dict] = []

        for processed, matched, result in _process_entries_parallel(
            entries,
            whitelists,
            scraper_config,
            session,
            llm_client,
            interests_text,
            product_config=config,
        ):
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

        _enrich_results_with_citations(results, session, config, now=now)

        _sort_results(results)
        new_count, skipped = _save_results(app, results)

        _emit(event_callback, "status", {"phase": "thumbnails", "message": "Generating PDF thumbnails..."})
        _generate_thumbnails(app, results, session)

        summary = _build_summary(new_count, skipped + pre_filtered, len(results), total_entries)
        _emit(event_callback, "done", summary)
        _finish_scrape_run(app, scrape_run_id, status="success")

        LOGGER.info(
            "Scrape complete: %s new, %s duplicates, %s matched out of %s entries",
            new_count,
            skipped + pre_filtered,
            len(results),
            total_entries,
        )
        return summary
    except Exception:
        _finish_scrape_run(app, scrape_run_id, status="error")
        raise


def run_scrape(app) -> dict:
    return execute_scrape(app, event_callback=None)


def stream_or_start_scrape(app, force: bool = False):
    """Compatibility wrapper implemented in job manager module."""
    from app.services.jobs import SCRAPE_JOB_MANAGER

    return SCRAPE_JOB_MANAGER.stream_for_request(app, force=force)


def execute_historical_scrape(app, categories: list[str], start_dt: date, end_dt: date) -> dict:
    from app.services.enrichment import query_arxiv_api, enrich_entries_with_api_metadata
    from app.services.http_client import create_session

    config = app.config["SCRAPER_CONFIG"]
    whitelists = config["whitelists"]
    scraper_config = config["scraper"]
    max_workers = max(1, int(scraper_config.get("max_workers", DEFAULT_MAX_WORKERS)))
    session = create_session(pool_size=max_workers)

    entries = query_arxiv_api(categories, start_dt, end_dt, max_results=2000)
    total_entries = len(entries)
    if not entries:
        return _build_summary(0, 0, 0, 0)
        
    existing_ids = _get_existing_ids(app, entries)
    pre_filtered = 0
    if existing_ids:
        filtered_entries = [e for e in entries if not _identity_keys(e).intersection(existing_ids)]
        pre_filtered = len(entries) - len(filtered_entries)
        entries = filtered_entries

    llm_client, interests_text = _create_llm_client(app)
    enrich_entries_with_api_metadata(entries, session=session)

    results = []
    for processed, matched, result in _process_entries_parallel(entries, whitelists, scraper_config, session, llm_client, interests_text, config):
        if result:
            results.append(result)

    _enrich_results_with_citations(results, session, config)

    _sort_results(results)
    new_count, skipped = _save_results(app, results)
    
    _generate_thumbnails(app, results, session)

    return _build_summary(new_count, skipped + pre_filtered, len(results), total_entries)

