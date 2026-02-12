import email.utils
import io
import json
import logging
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache

import PyPDF2
import feedparser
import requests
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
    )

MATCH_PRIORITY = {
    "Author": 1,
    "Affiliation": 2,
    "Title": 3,
}


def _normalize(text):
    """Strip accents so terms like 'Dollár' match 'Dollar'."""
    nfkd = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in nfkd if not unicodedata.combining(char))


def _dedupe_preserve_order(items):
    return list(dict.fromkeys(items))


def _build_pattern(term):
    """
    Build a regex source for a whitelist term.

    - ALL-CAPS short terms (<=4 chars, e.g. MIT/SAR) are case-sensitive
    - Multi-word terms accept hyphen/space/newline separators
    - Everything else is case-insensitive
    """
    normalized = _normalize(term)
    escaped = re.escape(normalized)
    is_short_acronym = len(term) <= 4 and term.isupper()
    flags = 0 if is_short_acronym else re.IGNORECASE

    if " " in term:
        flexible_spacing = escaped.replace(r"\ ", r"[-\s]+")
        return rf"\b{flexible_spacing}\b", flags

    return rf"\b{escaped}\b", flags


@lru_cache(maxsize=32)
def _compile_patterns(terms, mode):
    """Compile regex patterns once per process for a given whitelist tuple."""
    compiled = []

    if mode == "author":
        for term in terms:
            normalized = _normalize(term)
            pattern = re.compile(rf"\b{re.escape(normalized)}\b", re.IGNORECASE)
            compiled.append((term, pattern))
        return tuple(compiled)

    for term in terms:
        source, flags = _build_pattern(term)
        compiled.append((term, re.compile(source, flags)))

    return tuple(compiled)


def check_whitelist_match(text_list, whitelist):
    """Return deduplicated whitelist terms matched in any input text."""
    matches = []
    normalized_texts = [_normalize(text) for text in text_list if text]
    patterns = _compile_patterns(tuple(whitelist), mode="general")

    for term, pattern in patterns:
        if any(pattern.search(text) for text in normalized_texts):
            matches.append(term)

    return _dedupe_preserve_order(matches)


def check_author_match(author_names, whitelist):
    """Author-specific matcher that handles full names and last-name-only terms."""
    matches = []
    normalized_names = [_normalize(name.strip()) for name in author_names if name]
    patterns = _compile_patterns(tuple(whitelist), mode="author")

    for term, pattern in patterns:
        if any(pattern.search(name) for name in normalized_names):
            matches.append(term)

    return _dedupe_preserve_order(matches)


def _extract_affiliation_text(pdf_bytes, lines_start=2, lines_end=30):
    """Extract candidate affiliation text from the first PDF page."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        if not pdf_reader.pages:
            return ""

        page_text = pdf_reader.pages[0].extract_text() or ""
        lines = page_text.splitlines()
        return "\n".join(lines[lines_start:lines_end])
    except Exception:
        return ""


def _format_publication_date(published):
    if not published:
        return "Date Unknown"

    try:
        parsed = email.utils.parsedate_to_datetime(published)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return "Date Unknown"


def process_paper(entry_data, whitelists, scraper_config):
    """Process one entry: download PDF, match all categories, return metadata."""
    link = entry_data["link"]
    title = entry_data["title"]
    author_str = entry_data["author"]
    authors_list = entry_data["authors_list"]
    abstract = entry_data.get("abstract", "")
    publication_date = _format_publication_date(entry_data.get("published"))

    pdf_url = link.replace("/abs/", "/pdf/")

    try:
        pdf_response = requests.get(pdf_url, timeout=30)
        pdf_response.raise_for_status()
    except Exception as exc:
        LOGGER.error("Error fetching PDF for %s: %s", link, exc)
        return None

    affiliation_text = _extract_affiliation_text(
        pdf_response.content,
        lines_start=scraper_config.get("pdf_lines_start", 2),
        lines_end=scraper_config.get("pdf_lines_end", 30),
    )

    category_matches = {
        "Author": check_author_match(authors_list, whitelists["authors"]),
        "Affiliation": check_whitelist_match(
            [affiliation_text], whitelists["affiliations"]
        ),
        "Title": check_whitelist_match([title, abstract], whitelists["titles"]),
    }

    match_types = [name for name, terms in category_matches.items() if terms]
    if not match_types:
        return None

    all_terms = _dedupe_preserve_order(
        term for terms in category_matches.values() for term in terms
    )

    return {
        "title": title,
        "authors": author_str,
        "link": link,
        "pdf_link": pdf_url,
        "matches": all_terms,
        "match_type": " + ".join(match_types),
        "match_priority": min(MATCH_PRIORITY[name] for name in match_types),
        "publication_date": publication_date,
    }


def _extract_author_names(entry):
    if hasattr(entry, "authors") and entry.authors:
        names = [author.get("name", "") for author in entry.authors if author.get("name")]
        if names:
            return names

    raw_authors = getattr(entry, "author", "")
    return [name for name in re.split(r",\s*|\s+and\s+", raw_authors) if name]


def _clean_abstract(summary):
    if not summary:
        return ""
    return re.sub(r"<[^>]+>", " ", summary)


def _serialize_entries(feed):
    """Convert feedparser entries to plain dicts (pickle-friendly for workers)."""
    return [
        {
            "link": entry.link,
            "title": entry.title,
            "author": getattr(entry, "author", ""),
            "authors_list": _extract_author_names(entry),
            "abstract": _clean_abstract(getattr(entry, "summary", "")),
            "published": getattr(entry, "published", None),
        }
        for entry in feed.entries
    ]


def _process_entries_parallel(entries, whitelists, scraper_config):
    """Yield `(processed, matched, result)` as futures complete."""
    max_workers = scraper_config.get("max_workers", 8)
    processed = 0
    matched = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_paper, entry, whitelists, scraper_config)
            for entry in entries
        ]

        for future in as_completed(futures):
            processed += 1
            result = None
            try:
                result = future.result()
            except Exception:
                LOGGER.exception("Unhandled worker exception while processing paper")

            if result:
                matched += 1

            yield processed, matched, result


def _sort_results(results):
    results.sort(key=lambda item: (item["match_priority"], item["publication_date"]))


def _save_results(app, results):
    from app.models import Paper, db

    current_date = datetime.now().strftime("%Y-%m-%d")
    new_count = 0
    skipped = 0

    with app.app_context():
        links = [result["link"] for result in results]
        existing_links = set()

        if links:
            existing_rows = db.session.query(Paper.link).filter(Paper.link.in_(links)).all()
            existing_links = {link for (link,) in existing_rows}

        for result in results:
            if result["link"] in existing_links:
                skipped += 1
                continue

            db.session.add(
                Paper(
                    title=result["title"],
                    authors=result["authors"],
                    link=result["link"],
                    pdf_link=result["pdf_link"],
                    match_type=result["match_type"],
                    matched_terms=", ".join(result["matches"]),
                    publication_date=result["publication_date"],
                    scraped_date=current_date,
                )
            )
            new_count += 1

        if new_count:
            db.session.commit()

    return new_count, skipped


def _get_scrape_config(app):
    config = app.config["SCRAPER_CONFIG"]
    return config["whitelists"], config["scraper"]


def _parse_feed_entries(feed_url):
    feed = feedparser.parse(feed_url)
    entries = _serialize_entries(feed)
    LOGGER.info("Total entries in RSS feed: %s", len(entries))
    return entries


def _build_summary(new_count, skipped, total_matched, total_in_feed):
    return {
        "new_papers": new_count,
        "duplicates_skipped": skipped,
        "total_matched": total_matched,
        "total_in_feed": total_in_feed,
    }


def run_scrape(app):
    """Fetch, process in parallel, persist matches, and return a scrape summary."""
    whitelists, scraper_config = _get_scrape_config(app)
    entries = _parse_feed_entries(scraper_config["feed_url"])

    results = []
    with tqdm(total=len(entries), desc="Processing papers") as progress:
        for _, _, result in _process_entries_parallel(entries, whitelists, scraper_config):
            progress.update(1)
            if result:
                results.append(result)

    _sort_results(results)
    new_count, skipped = _save_results(app, results)

    LOGGER.info(
        "Scrape complete: %s new, %s duplicates, %s total matched out of %s entries",
        new_count,
        skipped,
        len(results),
        len(entries),
    )

    return _build_summary(new_count, skipped, len(results), len(entries))


def _sse_event(event, data):
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def run_scrape_stream(app):
    """Streaming variant of `run_scrape` that emits SSE progress events."""
    whitelists, scraper_config = _get_scrape_config(app)

    yield _sse_event("status", {"phase": "feed", "message": "Fetching RSS feed..."})
    entries = _parse_feed_entries(scraper_config["feed_url"])
    total = len(entries)

    yield _sse_event("feed", {"total": total})
    yield _sse_event(
        "status",
        {"phase": "processing", "message": f"Processing {total} papers..."},
    )

    results = []
    for processed, matched, result in _process_entries_parallel(
        entries,
        whitelists,
        scraper_config,
    ):
        if result:
            results.append(result)
            yield _sse_event(
                "match",
                {
                    "processed": processed,
                    "total": total,
                    "matched": matched,
                    "paper": {
                        "title": result["title"],
                        "match_type": result["match_type"],
                        "matched_terms": result["matches"],
                    },
                },
            )
            continue

        yield _sse_event(
            "progress",
            {"processed": processed, "total": total, "matched": matched},
        )

    yield _sse_event("status", {"phase": "saving", "message": "Saving to database..."})
    _sort_results(results)
    new_count, skipped = _save_results(app, results)

    yield _sse_event("done", _build_summary(new_count, skipped, len(results), total))
