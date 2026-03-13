"""Feed parsing and metadata enrichment helpers."""

from __future__ import annotations

import email.utils
import io
import logging
import re
import time
import defusedxml.ElementTree as ET
from datetime import date, datetime, timezone

import feedparser
import pdfplumber

from app.services.http_client import request_with_backoff
from app.services.text import clean_whitespace

LOGGER = logging.getLogger(__name__)

_HEADER_END_RE = re.compile(
    r"^\s*(?:Abstract|ABSTRACT|1[\.\s]+Introduction|I\.\s+Introduction)\b",
    re.MULTILINE,
)
_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/(.+?)(?:v\d+)?$")
_URL_RE = re.compile(r"https?://[^\s<>)\]\"']+")

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
from app.constants import ARXIV_API_BATCH_SIZE as _ARXIV_API_BATCH_SIZE
from app.constants import ARXIV_API_DELAY as _ARXIV_API_DELAY

_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def extract_arxiv_id(link: str) -> str | None:
    match = _ARXIV_ID_RE.search(link)
    return match.group(1) if match else None


def parse_publication_dt(published: str | None) -> tuple[date | None, str]:
    if not published:
        return None, "Date Unknown"

    try:
        parsed = email.utils.parsedate_to_datetime(published)
        parsed_date = parsed.date()
        return parsed_date, parsed_date.isoformat()
    except Exception:
        return None, "Date Unknown"


def clean_abstract(summary: str | None) -> str:
    if not summary:
        return ""
    no_html = re.sub(r"<[^>]+>", " ", summary)
    return clean_whitespace(no_html)


def extract_author_names(entry: feedparser.FeedParserDict) -> list[str]:
    if hasattr(entry, "authors") and entry.authors:
        names = [author.get("name", "") for author in entry.authors if author.get("name")]
        if names:
            return names

    raw_authors = getattr(entry, "author", "")
    return [name for name in re.split(r",\s*|\s+and\s+", raw_authors) if name]


def parse_feed_entries(feed_url: str) -> list[dict]:
    response = request_with_backoff("GET", feed_url, timeout=30)
    feed = feedparser.parse(response.content)

    entries = []
    for entry in feed.entries:
        link = entry.link
        publication_dt, publication_date = parse_publication_dt(getattr(entry, "published", None))

        entries.append(
            {
                "arxiv_id": extract_arxiv_id(link),
                "link": link,
                "title": clean_whitespace(entry.title),
                "author": getattr(entry, "author", ""),
                "authors_list": extract_author_names(entry),
                "abstract": clean_abstract(getattr(entry, "summary", "")),
                "published": getattr(entry, "published", None),
                "publication_dt": publication_dt,
                "publication_date": publication_date,
            }
        )

    LOGGER.info("Total entries in RSS feed: %s", len(entries))
    return entries


def _categorize_resource(url: str) -> tuple[str, str]:
    normalized = url.lower()
    if "github.com" in normalized or "gitlab.com" in normalized:
        return "code", "Code"
    if "huggingface.co/datasets" in normalized:
        return "dataset", "Dataset"
    if "huggingface.co/spaces" in normalized:
        return "demo", "Demo"
    if "/project/" in normalized or normalized.endswith("/project"):
        return "project", "Project"
    if "youtube.com" in normalized or "youtu.be" in normalized:
        return "video", "Video"
    return "web", "Link"


def extract_resource_links(*texts: str | None) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen: set[str] = set()

    for text in texts:
        if not text:
            continue
        for url in _URL_RE.findall(text):
            cleaned = url.rstrip(".,;:")
            if cleaned in seen:
                continue
            seen.add(cleaned)
            resource_type, label = _categorize_resource(cleaned)
            links.append({"type": resource_type, "label": label, "url": cleaned})

    return links


def _fetch_api_metadata(arxiv_ids: list[str]) -> dict[str, dict]:
    metadata: dict[str, dict] = {}

    for index in range(0, len(arxiv_ids), _ARXIV_API_BATCH_SIZE):
        if index > 0:
            time.sleep(_ARXIV_API_DELAY)

        batch = arxiv_ids[index : index + _ARXIV_API_BATCH_SIZE]
        params = {"id_list": ",".join(batch), "max_results": len(batch)}

        try:
            response = request_with_backoff(
                "GET",
                _ARXIV_API_URL,
                params=params,
                timeout=30,
                attempts=3,
                base_delay=1.5,
            )
        except Exception as exc:
            LOGGER.warning("arXiv API batch request failed: %s", exc)
            continue

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            LOGGER.warning("arXiv API XML parse error: %s", exc)
            continue

        for entry in root.findall("atom:entry", _ATOM_NS):
            id_el = entry.find("atom:id", _ATOM_NS)
            if id_el is None or not id_el.text:
                continue

            id_match = re.search(r"abs/(.+?)(?:v\d+)?$", id_el.text)
            if not id_match:
                continue

            arxiv_id = id_match.group(1)

            affiliations: list[str] = []
            for author in entry.findall("atom:author", _ATOM_NS):
                for affil in author.findall("arxiv:affiliation", _ATOM_NS):
                    if affil.text and affil.text.strip():
                        affiliations.append(affil.text.strip())

            categories = [
                cat.get("term", "").strip()
                for cat in entry.findall("atom:category", _ATOM_NS)
                if cat.get("term", "").strip()
            ]

            comment_el = entry.find("arxiv:comment", _ATOM_NS)
            doi_el = entry.find("arxiv:doi", _ATOM_NS)

            metadata[arxiv_id] = {
                "api_affiliations": "\n".join(dict.fromkeys(affiliations)),
                "categories": list(dict.fromkeys(categories)),
                "comment": comment_el.text.strip() if comment_el is not None and comment_el.text else "",
                "doi": doi_el.text.strip() if doi_el is not None and doi_el.text else "",
            }

    return metadata


def enrich_entries_with_api_metadata(entries: list[dict]) -> None:
    arxiv_ids = [entry["arxiv_id"] for entry in entries if entry.get("arxiv_id")]
    if not arxiv_ids:
        return

    LOGGER.info("Querying arXiv API metadata for %d papers...", len(arxiv_ids))
    metadata = _fetch_api_metadata(arxiv_ids)

    enriched = 0
    for entry in entries:
        arxiv_id = entry.get("arxiv_id")
        if not arxiv_id:
            continue
        data = metadata.get(arxiv_id)
        if not data:
            continue

        entry["api_affiliations"] = data.get("api_affiliations", "")
        entry["categories"] = data.get("categories", [])
        entry["comment"] = data.get("comment", "")
        entry["doi"] = data.get("doi", "")

        entry["resource_links"] = extract_resource_links(
            entry.get("abstract", ""),
            data.get("comment", ""),
            data.get("doi", ""),
        )
        enriched += 1

    LOGGER.info("arXiv API metadata enriched %d/%d papers", enriched, len(arxiv_ids))


def extract_affiliation_text(
    pdf_bytes: bytes,
    *,
    lines_start: int = 2,
    max_header_lines: int = 50,
    smart_header: bool = True,
) -> str:
    """Extract first-page text region that usually includes author affiliations."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                return ""
            page_text = pdf.pages[0].extract_text() or ""
    except Exception as exc:
        LOGGER.warning("Failed to extract affiliations from PDF: %s", exc)
        return ""

    if smart_header:
        match = _HEADER_END_RE.search(page_text)
        if match:
            page_text = page_text[: match.start()]

    lines = page_text.splitlines()
    return "\n".join(lines[lines_start:max_header_lines])


def now_utc() -> datetime:
    # Kept for backward compatibility; canonical version is in app.services.text.
    from app.services.text import now_utc as _now_utc

    return _now_utc()
