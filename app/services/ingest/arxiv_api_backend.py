"""arXiv API ingest backend, the one entry point to arXiv's export API, and its OAI-PMH fallback."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from typing import Any

import defusedxml.ElementTree as ET
import requests

from app.constants import ARXIV_API_BATCH_SIZE as _ARXIV_API_BATCH_SIZE
from app.constants import ARXIV_API_DELAY as _ARXIV_API_DELAY
from app.services.http_client import request_with_backoff
from app.services.ingest.base import PaperCandidate, clean_abstract, extract_arxiv_id, parse_publication_dt
from app.services.text import clean_whitespace

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[int, PaperCandidate], None]

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ARXIV_API_TIMEOUT = 45
_ARXIV_API_ATTEMPTS = 4
_ARXIV_API_BASE_DELAY = 2.0
_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# A throttled host gets 403/406 (empty body) on every uncached export-API request, or 429s
# that outlive the retries, and waiting minutes does not clear it. Remember the refusal so
# every caller goes straight to its fallback instead of re-asking.
_REFUSED_STATUSES = frozenset({403, 406, 429})
_REFUSAL_MEMO_SECONDS = 30 * 60
# ponytail: in-process memo, (monotonic deadline, status). The cron runs one process per day
# and the web app is one long-lived process, so no cross-process file; add one if several
# short-lived processes start re-asking arXiv.
_refused: tuple[float, int] = (0.0, 0)

_OAI_URL = "https://oaipmh.arxiv.org/oai"
_OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "arxiv": "http://arxiv.org/OAI/arXiv/"}
# ~1 request/s through the shared token bucket, for GetRecord's one-request-per-id loop.
_OAI_PACE = {"ingest": {"rate_limit": {"requests_per_second": 1.0, "burst": 1}}}
# ponytail: GetRecord is one request per id, so one call resolves at most this many (~30 s,
# which keeps the synchronous import-ids route bearable); the rest come back unresolved and
# a retry picks them up, since callers only look up ids they don't store yet. Upgrade:
# Semantic Scholar's batch endpoint (500 ids per request, no arXiv categories).
_OAI_LOOKUP_CAP = 25
# A ListRecords page holds ~1300 records (~4 MB).
_OAI_MAX_PAGES = 10
_OAI_GROUPS = frozenset({"cs", "econ", "eess", "math", "q-bio", "q-fin", "stat"})


class ArxivRefused(requests.HTTPError):
    """arXiv's export API refused this host (403/406, or 429 once the retries ran out)."""

    def __init__(self, status: int, response: requests.Response | None = None):
        super().__init__(f"arXiv refused the request (HTTP {status})", response=response)
        self.status = status


def request_arxiv_api(params: dict[str, str | int], **kwargs: Any) -> requests.Response:
    """GET the export API. Every caller comes through here, so one refusal pauses them all."""
    global _refused
    until, status = _refused
    if time.monotonic() < until:
        raise ArxivRefused(status)
    try:
        return request_with_backoff(
            "GET",
            _ARXIV_API_URL,
            params=params,
            timeout=_ARXIV_API_TIMEOUT,
            attempts=_ARXIV_API_ATTEMPTS,
            base_delay=_ARXIV_API_BASE_DELAY,
            rate_limit_profile="bulk",
            **kwargs,
        )
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status not in _REFUSED_STATUSES:
            raise
        _refused = (time.monotonic() + _REFUSAL_MEMO_SECONDS, status)
        LOGGER.warning("arXiv export API refused this host (HTTP %s); using fallbacks for 30 minutes", status)
        raise ArxivRefused(status, response=exc.response) from exc


def _oai_root(params: dict[str, str], user_agent: str | None = None) -> ET.Element:
    # No session: its limiter/UA would be retuned to _OAI_PACE. The caller passes its UA.
    response = request_with_backoff(
        "GET",
        _OAI_URL,
        params=params,
        timeout=_ARXIV_API_TIMEOUT,
        attempts=_ARXIV_API_ATTEMPTS,
        base_delay=_ARXIV_API_BASE_DELAY,
        scraper_config=_OAI_PACE,
        user_agent=user_agent,
        max_bytes=25 * 1024 * 1024,
    )
    return ET.fromstring(response.content)


def _parse_oai_record(meta: ET.Element) -> PaperCandidate:
    """Map an OAI-PMH ``arXiv``-format record onto the candidate shape the Atom API yields.

    ``<created>`` stands in for the Atom ``<published>`` (v1) date, but arXiv's OAI feed
    reports a later version's date there for revised papers.
    """

    def text(tag: str) -> str:
        return clean_whitespace(meta.findtext(f"arxiv:{tag}", "", _OAI_NS))

    authors: list[str] = []
    affiliations: list[str] = []
    for author in meta.iterfind("arxiv:authors/arxiv:author", _OAI_NS):
        names = (author.findtext(f"arxiv:{tag}", "", _OAI_NS) for tag in ("forenames", "keyname", "suffix"))
        authors.append(clean_whitespace(" ".join(names)))
        affiliations += [clean_whitespace(el.text or "") for el in author.iterfind("arxiv:affiliation", _OAI_NS)]
    link = f"https://arxiv.org/abs/{text('id')}"
    created = text("created")
    publication_dt, publication_date = parse_publication_dt(created or None)
    return PaperCandidate(
        arxiv_id=extract_arxiv_id(link),
        link=link,
        title=text("title"),
        author=", ".join(authors),
        authors_list=authors,
        abstract=clean_abstract(meta.findtext("arxiv:abstract", "", _OAI_NS)),
        published=created or None,
        publication_dt=publication_dt,
        publication_date=publication_date,
        categories=text("categories").split(),
        comment=text("comments"),
        doi=text("doi"),
        api_affiliations="\n".join(dict.fromkeys(filter(None, affiliations))),
    )


def fetch_oai_records(arxiv_ids: Sequence[str], user_agent: str | None = None) -> dict[str, PaperCandidate]:
    """Look ids up via OAI-PMH GetRecord, the fallback when the export API refuses us.

    Best effort, like the id_list query: unknown ids are absent from the result, and a
    request that still fails after its retries ends the loop (OAI is struggling too).
    """
    if len(arxiv_ids) > _OAI_LOOKUP_CAP:
        LOGGER.warning("arXiv OAI-PMH fallback resolves %d of %d ids this call", _OAI_LOOKUP_CAP, len(arxiv_ids))
    found: dict[str, PaperCandidate] = {}
    for arxiv_id in arxiv_ids[:_OAI_LOOKUP_CAP]:
        params = {"verb": "GetRecord", "identifier": f"oai:arXiv.org:{arxiv_id}", "metadataPrefix": "arXiv"}
        try:
            root = _oai_root(params, user_agent)
        except Exception as exc:
            LOGGER.warning("arXiv OAI-PMH lookup failed at %s; stopping the fallback: %s", arxiv_id, exc)
            break
        for meta in root.iterfind(".//arxiv:arXiv", _OAI_NS):
            candidate = _parse_oai_record(meta)
            if candidate.arxiv_id:
                found[candidate.arxiv_id] = candidate
    return found


def _oai_set(category: str) -> str:
    """cs.CV -> cs:cs:CV, astro-ph.CO -> physics:astro-ph:CO, hep-th -> physics:hep-th."""
    archive, _, subject = category.partition(".")
    group = archive if archive in _OAI_GROUPS else "physics"
    return ":".join(part for part in (group, archive, subject) if part)


def list_oai_candidates(
    categories: Sequence[str], start_dt: date, end_dt: date, max_results: int, user_agent: str | None = None
) -> list[PaperCandidate]:
    """List a submission window via OAI-PMH ListRecords, the fallback when the export API refuses us.

    OAI datestamps are last-modified, so this harvests from ``start_dt`` with no ``until`` (a
    paper submitted in the window and revised since carries a later datestamp) and keeps the
    records whose ``<created>`` falls in the window. ``<created>`` can be a revision's date, so
    the id's YYMM must also fall in the window (plus a month of announcement lag); that drops
    old papers revised in the window. Newest first and capped, like the API listing.

    ponytail: the cost is every record revised since ``start_dt`` — fine for the rolling
    window, catch-up and recent sync chunks; a deeper window raises past _OAI_MAX_PAGES rather
    than return a silent gap, and a paper revised after its window closed is missed
    (end = today loses nothing). Upgrade: arXiv's bulk metadata snapshot for deep backfills.
    """
    first_month = start_dt.strftime("%y%m")
    last_month = (end_dt.replace(day=28) + timedelta(days=7)).strftime("%y%m")
    found: dict[str, PaperCandidate] = {}
    for category in categories:
        LOGGER.warning("Listing %s since %s via arXiv OAI-PMH", category, start_dt)
        params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": _oai_set(category), "from": str(start_dt)}
        for _page in range(_OAI_MAX_PAGES):
            root = _oai_root(params, user_agent)
            error = root.find("oai:error", _OAI_NS)
            if error is not None and error.get("code") != "noRecordsMatch":
                raise RuntimeError(f"arXiv OAI-PMH error for {category}: {error.get('code')} {error.text or ''}")
            for meta in root.iterfind(".//arxiv:arXiv", _OAI_NS):
                candidate = _parse_oai_record(meta)
                aid = candidate.arxiv_id or ""
                in_window = candidate.publication_dt is not None and start_dt <= candidate.publication_dt <= end_dt
                if in_window and (not aid[:4].isdigit() or first_month <= aid[:4] <= last_month):
                    found[aid] = candidate
            token = root.findtext(".//oai:resumptionToken", "", _OAI_NS).strip()
            if not token:
                break
            params = {"verb": "ListRecords", "resumptionToken": token}
        else:
            raise RuntimeError(
                f"arXiv OAI-PMH listing of {category} since {start_dt} exceeds {_OAI_MAX_PAGES} pages; "
                "narrow the window or retry once the arXiv API accepts requests again"
            )
    newest_first = sorted(found.values(), key=lambda c: (c.publication_date, c.arxiv_id or ""), reverse=True)
    return newest_first[:max_results]


def _build_query(categories: Sequence[str], start_dt: date, end_dt: date, query: str | None = None) -> str:
    """AND an optional category filter, an optional raw arXiv search query and the date window."""
    clauses = []
    if categories:
        cat_query = " OR ".join(f"cat:{category}" for category in categories)
        clauses.append(f"({cat_query})")
    if query:
        clauses.append(f"({query})")
    from_ts = start_dt.strftime("%Y%m%d0000")
    to_ts = end_dt.strftime("%Y%m%d2359")
    clauses.append(f"submittedDate:[{from_ts} TO {to_ts}]")
    return " AND ".join(clauses)


def _parse_atom_candidate(entry: ET.Element) -> PaperCandidate:
    id_el = entry.find("atom:id", _ATOM_NS)
    title_el = entry.find("atom:title", _ATOM_NS)
    summary_el = entry.find("atom:summary", _ATOM_NS)
    published_el = entry.find("atom:published", _ATOM_NS)
    comment_el = entry.find("arxiv:comment", _ATOM_NS)
    doi_el = entry.find("arxiv:doi", _ATOM_NS)

    link = clean_whitespace(id_el.text if id_el is not None and id_el.text else "")
    authors_list = [
        clean_whitespace(name_el.text)
        for author_el in entry.findall("atom:author", _ATOM_NS)
        for name_el in [author_el.find("atom:name", _ATOM_NS)]
        if name_el is not None and name_el.text
    ]
    categories = [
        term
        for term in (category_el.get("term", "").strip() for category_el in entry.findall("atom:category", _ATOM_NS))
        if term
    ]
    published = clean_whitespace(published_el.text if published_el is not None and published_el.text else "")
    publication_dt, publication_date = parse_publication_dt(published or None)

    return PaperCandidate(
        arxiv_id=extract_arxiv_id(link),
        link=link,
        title=clean_whitespace(title_el.text if title_el is not None else ""),
        author=", ".join(authors_list),
        authors_list=authors_list,
        abstract=clean_abstract(summary_el.text if summary_el is not None else ""),
        published=published or None,
        publication_dt=publication_dt,
        publication_date=publication_date,
        categories=categories,
        comment=clean_whitespace(comment_el.text if comment_el is not None else ""),
        doi=clean_whitespace(doi_el.text if doi_el is not None else ""),
    )


class ArxivApiBackend:
    def __init__(self, *, page_size: int = _ARXIV_API_BATCH_SIZE, delay_seconds: float = _ARXIV_API_DELAY):
        self.page_size = page_size
        self.delay_seconds = delay_seconds

    @property
    def name(self) -> str:
        return "arxiv_api"

    def fetch(
        self,
        *,
        categories: Sequence[str],
        start_dt: date,
        end_dt: date,
        max_results: int = 1000,
        session: requests.Session | None = None,
        offset: int = 0,
        resume_after_arxiv_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
        user_agent: str | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> list[PaperCandidate]:
        del kwargs

        if not (categories or query) or max_results <= 0:
            return []

        query_str = _build_query(categories, start_dt, end_dt, query)
        results: list[PaperCandidate] = []
        start = max(0, int(offset))
        resume_page = ((start // self.page_size) + 1) if resume_after_arxiv_id else None
        resume_consumed = resume_after_arxiv_id is None

        while len(results) < max_results:
            if start > max(0, int(offset)) and self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

            batch_limit = min(self.page_size, max_results - len(results))
            try:
                response = request_arxiv_api(
                    {
                        "search_query": query_str,
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                        "start": start,
                        "max_results": batch_limit,
                    },
                    session=session,
                    user_agent=user_agent,
                    # An Atom page of <= page_size entries is a few MB; cap well below the
                    # global default so a hostile response can't buffer 200 MB of XML.
                    max_bytes=25 * 1024 * 1024,
                )
            except ArxivRefused:
                if query:
                    raise  # a search has no OAI equivalent; cv-arxiv-sync --query asks Semantic Scholar
                # The whole window at once: the page cursor (offset/resume/progress) is API-only.
                return list_oai_candidates(categories, start_dt, end_dt, max_results, user_agent)
            root = ET.fromstring(response.text)
            entries = root.findall("atom:entry", _ATOM_NS)
            if not entries:
                break

            candidates = [_parse_atom_candidate(entry) for entry in entries]

            # If the saved cursor is no longer present on the page we resumed
            # from, it was pushed off by submissions/withdrawals that landed
            # between runs. Skipping the page (waiting to re-find the cursor)
            # would silently drop every paper on it, so treat a missing cursor as
            # "already past" and resume consuming from the page's first entry.
            # Re-processing a few already-seen papers is harmless — _save_results
            # de-dupes on the unique arxiv_id.
            if (
                not resume_consumed
                and resume_page is not None
                and resume_after_arxiv_id not in {c.arxiv_id for c in candidates}
            ):
                resume_consumed = True

            for batch_index, candidate in enumerate(candidates):
                current_page = ((start + batch_index) // self.page_size) + 1

                if not resume_consumed and resume_page is not None:
                    if current_page > resume_page:
                        resume_consumed = True
                    elif candidate.arxiv_id == resume_after_arxiv_id:
                        resume_consumed = True
                        continue
                    else:
                        continue

                results.append(candidate)
                if progress_callback is not None:
                    progress_callback(current_page, candidate)
                if len(results) >= max_results:
                    break

            if len(entries) < batch_limit:
                break
            start += batch_limit

        return results
