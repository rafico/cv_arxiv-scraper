"""Core scrape execution pipeline."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests
from sqlalchemy.exc import IntegrityError

from app.constants import DEFAULT_LLM_MODEL, DEFAULT_MAX_WORKERS
from app.services.enrichment import (
    enrich_entries_with_api_metadata,
    extract_pdf_resource_links,
    fetch_recent_papers,
    merge_resource_links,
    parse_feed_entries,
)
from app.services.http_client import create_session, resolve_user_agent
from app.services.ingest import IngestMode, IngestOrchestrator, PaperCandidate
from app.services.interest_model import build_interest_profile
from app.services.llm_client import LLMClient, resolve_api_key
from app.services.pipeline import WeightedSumRanker, WhitelistCandidateGenerator
from app.services.preferences import get_preferences
from app.services.ranking import compute_paper_score
from app.services.summary import extract_topic_tags, generate_llm_summary, generate_summary
from app.services.text import now_utc

LOGGER = logging.getLogger(__name__)

# Wall-clock budget for an isolated native stage (embeddings / PDF render) before the
# child is terminated and the stage is skipped as non-fatal.
_NATIVE_STAGE_TIMEOUT = float(os.environ.get("CV_ARXIV_NATIVE_STAGE_TIMEOUT", "900"))


EventCallback = Callable[[str, dict], None] | None


def _build_ingest_orchestrator() -> IngestOrchestrator:
    return IngestOrchestrator(
        rss_candidate_fetcher=lambda feed_url, *, session=None: [
            PaperCandidate.from_entry_dict(entry) for entry in parse_feed_entries(feed_url, session=session)
        ],
        rolling_window_fetcher=lambda days, feed_url, *, session=None: [
            PaperCandidate.from_entry_dict(entry) for entry in fetch_recent_papers(days, feed_url, session=session)
        ],
    )


def _candidate_entries(candidates: list[PaperCandidate]) -> list[dict]:
    return [candidate.to_entry_dict() for candidate in candidates]


def _collect_feed_urls(app, scraper_config: dict) -> list[str]:
    feed_urls = list(scraper_config.get("feed_urls") or [])
    if scraper_config.get("feed_url") and scraper_config["feed_url"] not in feed_urls:
        feed_urls.append(scraper_config["feed_url"])

    try:
        from app.models import FeedSource

        with app.app_context():
            extra_sources = FeedSource.query.filter_by(enabled=True).all()
            for src in extra_sources:
                if src.url not in feed_urls:
                    feed_urls.append(src.url)
    except Exception:
        pass  # FeedSource table may not exist yet.

    return feed_urls


def _emit(callback: EventCallback, event: str, data: dict) -> None:
    if callback:
        callback(event, data)


def _enrich_candidate_with_llm(
    candidate,
    llm_client: LLMClient | None,
    interests_text: str,
    structured_insights: bool = False,
) -> None:
    """Add LLM summary, relevance score, and topic tags to candidate entry_data (in-place).

    With structured_insights enabled, one combined JSON call replaces the
    TLDR + relevance pair and additionally extracts tasks/datasets/method/
    backbone. Per-paper parse failures fall back to the legacy path.
    """
    entry = candidate.entry_data
    title = entry.get("title", "")
    abstract = entry.get("abstract", "")

    insights = None
    if llm_client is not None and structured_insights:
        insights = llm_client.analyze_paper(title, abstract, interests_text, matched_terms=candidate.matched_terms)

    if insights is not None:
        tldr = insights.get("tldr") or ""
        entry["summary_text"] = tldr[:280].rstrip() if tldr else generate_summary(title, abstract)
        entry["llm_relevance_score"] = insights.get("relevance")
        entry["llm_insights"] = {
            key: insights[key] for key in ("tasks", "datasets", "method_type", "backbone", "why_matched")
        }
    else:
        entry["summary_text"] = (
            generate_llm_summary(llm_client, title, abstract)
            if llm_client is not None
            else generate_summary(title, abstract)
        )
        entry["llm_relevance_score"] = (
            llm_client.rate_relevance(title, abstract, interests_text) if llm_client is not None else None
        )
        entry["llm_insights"] = {}
    entry["topic_tags"] = extract_topic_tags(title, abstract)


def _process_entries_with_pipeline(
    entries: list[dict],
    whitelists: dict,
    scraper_config: dict,
    session: requests.Session,
    llm_client: LLMClient | None = None,
    interests_text: str = "",
    product_config: dict | None = None,
    interest_profile=None,
):
    """Process entries using the ranking pipeline (candidates -> features -> rank).

    Yields (processed, matched, result_dict) tuples for streaming progress;
    result_dict is None for entries that did not match.
    """
    max_workers = max(1, int(scraper_config.get("max_workers", DEFAULT_MAX_WORKERS)))
    preferences = get_preferences(product_config)
    muted = preferences["muted"]
    structured_insights = bool(((product_config or {}).get("llm") or {}).get("structured_insights", False))

    generator = WhitelistCandidateGenerator(
        whitelists=whitelists,
        scraper_config=scraper_config,
        muted=muted,
        session=session,
    )
    ranker = WeightedSumRanker(config=product_config, interest_profile=interest_profile)

    processed = 0
    matched = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generator.process_single, entry): entry for entry in entries}

        for future in as_completed(futures):
            entry = futures[future]
            processed += 1
            candidate = None
            try:
                candidate = future.result()
            except Exception:
                LOGGER.exception(
                    "Unhandled worker exception while processing paper: %s (%s)",
                    entry.get("title"),
                    entry.get("link"),
                )

            if candidate is not None:
                _enrich_candidate_with_llm(candidate, llm_client, interests_text, structured_insights)
                ranked_list = ranker.rank([candidate])
                if ranked_list:
                    ranked = ranked_list[0]
                    result = ranked.to_result_dict()
                    matched += 1
                    yield processed, matched, result
                    continue

            yield processed, matched, None


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


def _filter_existing_entries(app, entries: list[dict]) -> tuple[list[dict], int]:
    """Drop entries already stored in the DB before doing any heavy work.

    Returns (remaining_entries, dropped_count).
    """
    existing_ids = _get_existing_ids(app, entries)
    if not existing_ids:
        return entries, 0

    remaining = [entry for entry in entries if not _identity_keys(entry).intersection(existing_ids)]
    pre_filtered = len(entries) - len(remaining)
    if pre_filtered:
        LOGGER.info(
            "Skipped %d already-stored papers, %d new to process",
            pre_filtered,
            len(remaining),
        )
    return remaining, pre_filtered


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
        papers_to_insert = []
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
                llm_insights=result.get("llm_insights", {}),
                arxiv_comment=result.get("arxiv_comment"),
                venue=result.get("venue"),
                venue_year=result.get("venue_year"),
                acceptance_status=result.get("acceptance_status"),
                interest_similarity=result.get("interest_similarity"),
                publication_date=result["publication_date"],
                publication_dt=result.get("publication_dt"),
                scraped_date=today_str,
                scraped_at=now,
                duplicate_of_id=duplicate_of_id,
                citation_count=result.get("citation_count"),
                influential_citation_count=result.get("influential_citation_count"),
                semantic_scholar_id=result.get("semantic_scholar_id"),
                citation_source=result.get("citation_source"),
                citation_provenance=result.get("citation_provenance", {}),
                citation_updated_at=result.get("citation_updated_at"),
                openalex_id=result.get("openalex_id"),
                openalex_topics=result.get("openalex_topics", []),
                oa_status=result.get("oa_status"),
                referenced_works_count=result.get("referenced_works_count"),
                openalex_cited_by_count=result.get("openalex_cited_by_count"),
            )
            papers_to_insert.append(paper)

        if papers_to_insert:
            try:
                db.session.add_all(papers_to_insert)
                db.session.commit()
                new_count += len(papers_to_insert)
            except IntegrityError:
                db.session.rollback()
                # Fallback to row-by-row insertion so one bad row does not drop the whole batch.
                for paper in papers_to_insert:
                    db.session.add(paper)
                    try:
                        db.session.commit()
                        new_count += 1
                    except IntegrityError as exc:
                        db.session.rollback()
                        skipped += 1
                        msg = str(exc.orig) if exc.orig is not None else str(exc)
                        if "UNIQUE" in msg.upper() or "unique constraint" in msg.lower():
                            LOGGER.debug("Skipping paper on unique conflict: %s (%s)", paper.link, msg)
                        else:
                            LOGGER.warning(
                                "Non-unique IntegrityError inserting paper %s: %s",
                                paper.link,
                                msg,
                            )

    return new_count, skipped


def _generate_thumbnails(app, results: list[dict], session: requests.Session) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    from app.services.thumbnail_generator import DEFAULT_THUMBNAIL_DPI, generate_thumbnail

    static_folder = app.static_folder if app.static_folder else Path(__file__).parent.parent / "static"
    scraper_config = app.config["SCRAPER_CONFIG"].get("scraper", {}) or {}
    resolution = int(scraper_config.get("thumbnail_dpi", DEFAULT_THUMBNAIL_DPI))

    def worker(res):
        arxiv_id = res.get("arxiv_id") or (res.get("link") or "").split("/")[-1]
        pdf_link = res.get("pdf_link")
        pdf_content = res.get("pdf_content")
        if arxiv_id and pdf_link:
            generate_thumbnail(
                arxiv_id, pdf_link, static_folder, session=session, pdf_content=pdf_content, resolution=resolution
            )

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Timeout after 120s to avoid blocking the gunicorn worker
            for _ in executor.map(worker, results, timeout=120):
                pass
    except FuturesTimeout:
        LOGGER.warning("Thumbnail generation timed out after 120s, skipping remaining")


def _generate_embeddings(app, results: list[dict]) -> None:
    """Generate SPECTER2 embeddings for newly scraped papers and add to the FAISS index."""
    try:
        from app.models import Paper
        from app.services.embeddings import add_papers_to_index, get_embedding_service, reset_embedding_service
        from app.services.subprocess_runner import run_isolated

        service = get_embedding_service(app)
        index_dir = str(service.index_dir)
        with app.app_context():
            paper_ids = []
            texts = []
            vectors = []
            for result in results:
                paper = Paper.query.filter_by(link=result["link"]).first()
                if paper and not service.has_paper(paper.id):
                    paper_ids.append(paper.id)
                    texts.append(f"{paper.title} {paper.abstract_text or ''}")
                    # Reuse the vector computed during interest scoring, if any.
                    vectors.append(result.get("embedding"))

        if paper_ids:
            # The faiss + torch work is the confirmed native-crash site; run it in a
            # child process so a SIGSEGV/abort there can't take down the server, then
            # reload the singleton from the index the child persisted.
            added = run_isolated(
                add_papers_to_index, index_dir, paper_ids, texts, vectors, timeout=_NATIVE_STAGE_TIMEOUT
            )
            reset_embedding_service()
            LOGGER.info("Generated embeddings for %d papers", added)
    except Exception:
        LOGGER.warning("Embedding generation failed (non-fatal)", exc_info=True)


def _extract_sections(app, results: list[dict]) -> None:
    """Optionally extract PDF sections and generate section-level embeddings."""
    scraper_config = app.config["SCRAPER_CONFIG"].get("scraper", {})
    if not scraper_config.get("extract_sections", False):
        return

    from app.models import Paper
    from app.services.pdf_extraction import extract_and_store_sections

    total_sections = 0
    with app.app_context():
        for result in results:
            pdf_content = result.get("pdf_content")
            if not pdf_content:
                continue
            paper = Paper.query.filter_by(link=result["link"]).first()
            if not paper:
                continue
            try:
                count = extract_and_store_sections(paper.id, pdf_content)
                total_sections += count
            except Exception:
                LOGGER.warning("Section extraction failed for %s", result.get("link"), exc_info=True)

    if total_sections > 0:
        LOGGER.info("Extracted %d sections from matched papers", total_sections)

        # Generate section-level embeddings.
        try:
            from app.models import PaperSection
            from app.services.embeddings import get_embedding_service

            service = get_embedding_service(app)
            with app.app_context():
                sections = PaperSection.query.join(Paper).filter(Paper.link.in_([r["link"] for r in results])).all()
                entries = [(s.paper_id, s.section_type, s.text) for s in sections if s.text]
                if entries:
                    added = service.add_sections(entries)
                    service.save_sections()
                    LOGGER.info("Generated section embeddings for %d sections", added)
        except Exception:
            LOGGER.warning("Section embedding generation failed (non-fatal)", exc_info=True)


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
        reasoning_effort = llm_config.get("reasoning_effort", "none")
    else:
        api_key = resolve_api_key(Path(app.config["LLM_KEY_PATH"]))
        if not api_key:
            LOGGER.warning("LLM is enabled but no API key is available")
            return None, ""
        default_base_url = "https://openrouter.ai/api/v1"
        default_model = DEFAULT_LLM_MODEL
        reasoning_effort = None

    try:
        client = LLMClient(
            api_key=api_key,
            model=llm_config.get("model", default_model),
            base_url=llm_config.get("base_url", default_base_url),
            max_concurrent=int(llm_config.get("max_concurrent", 4)),
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:
        LOGGER.warning("Unable to initialize LLM client: %s", exc)
        return None, ""

    interests_text = _build_llm_interests(app.config["SCRAPER_CONFIG"]["whitelists"])
    return client, interests_text


def _rescore_result(res: dict, config: dict | None) -> None:
    """Recompute the paper score after enrichment added new signals (in-place)."""
    res["paper_score"] = compute_paper_score(
        match_types=res.get("match_types", []),
        matched_terms_count=len(res.get("matches", [])),
        publication_dt=res.get("publication_dt"),
        resource_count=len(res.get("resource_links", [])),
        llm_relevance_score=res.get("llm_relevance_score"),
        citation_count=res.get("citation_count"),
        acceptance_status=res.get("acceptance_status"),
        interest_similarity=res.get("interest_similarity"),
        config=config,
    )


def _mark_citation_source(res: dict, source: str, now) -> None:
    res["citation_source"] = source
    res["citation_provenance"] = {
        "source": source,
        "updated_at": now.isoformat(),
    }
    res["citation_updated_at"] = now


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
                _mark_citation_source(res, "semantic_scholar", now)
            _rescore_result(res, config)


def _enrich_results_with_openalex(
    results: list[dict],
    session,
    config: dict,
) -> None:
    """Enrich matched results with OpenAlex metadata (in-place)."""
    openalex_config = config.get("openalex", {})
    if not openalex_config.get("enabled", True):
        return
    if not results:
        return

    from app.services.openalex import fetch_openalex_batch

    arxiv_ids = [res["arxiv_id"] for res in results if res.get("arxiv_id")]
    if not arxiv_ids:
        return

    email = openalex_config.get("email") or None
    openalex_data = fetch_openalex_batch(arxiv_ids, session=session, email=email)
    now = now_utc()
    for res in results:
        arxiv_id = res.get("arxiv_id")
        if arxiv_id and arxiv_id in openalex_data:
            data = openalex_data[arxiv_id]
            res["openalex_id"] = data.get("openalex_id")
            res["openalex_topics"] = data.get("openalex_topics", [])
            res["oa_status"] = data.get("oa_status")
            res["openalex_cited_by_count"] = data.get("openalex_cited_by_count")
            res["referenced_works_count"] = data.get("referenced_works_count")
            if res.get("citation_count") is None and res["openalex_cited_by_count"] is not None:
                res["citation_count"] = res["openalex_cited_by_count"]
                _mark_citation_source(res, "openalex", now)
                _rescore_result(res, config)


def _enrich_results_with_github(app, results: list[dict], session: requests.Session, config: dict) -> None:
    """Fetch GitHub repo metadata (stars, license) for saved papers with code links."""
    github_config = config.get("github", {})
    if not github_config.get("enabled", True):
        return

    import os

    from app.services.enrichment_providers import GitHubProvider, extract_github_repo

    repos_by_arxiv_id: dict[str, str] = {}
    for res in results:
        arxiv_id = res.get("arxiv_id")
        repo = extract_github_repo(res.get("resource_links"))
        if arxiv_id and repo:
            repos_by_arxiv_id[arxiv_id] = repo
    if not repos_by_arxiv_id:
        return

    token = os.environ.get("GITHUB_TOKEN") or github_config.get("token") or None
    provider = GitHubProvider(token=token)
    try:
        with app.app_context():
            payloads = provider.fetch_batch(
                list(repos_by_arxiv_id),
                repos_by_arxiv_id=repos_by_arxiv_id,
                session=session,
            )
            if not payloads:
                return

            from app.models import Paper, db

            papers = Paper.query.filter(Paper.arxiv_id.in_(list(payloads))).all()
            for paper in papers:
                data = payloads.get(paper.arxiv_id) or {}
                paper.github_repo = data.get("github_repo")
                paper.github_stars = data.get("github_stars")
                paper.github_license = data.get("github_license")
            db.session.commit()
    except Exception:
        LOGGER.warning("GitHub enrichment failed (non-fatal)", exc_info=True)


def _enrich_results_with_pdf_links(results: list[dict], config: dict | None) -> None:
    """Merge code/project links found in PDF front matter into resource_links (in-place)."""
    for res in results:
        # .get(), not .pop(): pdf_content is still needed by thumbnails/sections.
        pdf_links = extract_pdf_resource_links(res.get("pdf_content"))
        if not pdf_links:
            continue
        merged = merge_resource_links(res.get("resource_links"), pdf_links)
        if len(merged) != len(res.get("resource_links") or []):
            res["resource_links"] = merged
            _rescore_result(res, config)


def _collect_matched_results(
    entries: list[dict],
    whitelists: dict,
    scraper_config: dict,
    session: requests.Session,
    llm_client: LLMClient | None,
    interests_text: str,
    config: dict,
    *,
    total_entries: int,
    event_callback: EventCallback = None,
    interest_profile=None,
) -> list[dict]:
    """Run entries through the ranking pipeline, emitting progress events."""
    results: list[dict] = []
    for processed, matched, result in _process_entries_with_pipeline(
        entries,
        whitelists,
        scraper_config,
        session,
        llm_client,
        interests_text,
        product_config=config,
        interest_profile=interest_profile,
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
    return results


def _finalize_results(
    app,
    results: list[dict],
    session: requests.Session,
    config: dict,
    *,
    pre_filtered: int,
    total_entries: int,
    event_callback: EventCallback = None,
    now=None,
) -> dict:
    """Enrich, persist, and post-process matched results; returns the run summary."""
    _emit(event_callback, "status", {"phase": "saving", "message": "Saving to database..."})
    _enrich_results_with_citations(results, session, config, now=now)
    _enrich_results_with_openalex(results, session, config)
    _enrich_results_with_pdf_links(results, config)

    _sort_results(results)
    new_count, skipped = _save_results(app, results)
    _enrich_results_with_github(app, results, session, config)

    _emit(event_callback, "status", {"phase": "thumbnails", "message": "Generating PDF thumbnails..."})
    _generate_thumbnails(app, results, session)

    _emit(event_callback, "status", {"phase": "embeddings", "message": "Generating embeddings..."})
    _generate_embeddings(app, results)

    _emit(event_callback, "status", {"phase": "sections", "message": "Extracting PDF sections..."})
    _extract_sections(app, results)

    return _build_summary(new_count, skipped + pre_filtered, len(results), total_entries)


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

    session = None
    try:
        max_workers = max(1, int(scraper_config.get("max_workers", DEFAULT_MAX_WORKERS)))
        user_agent = resolve_user_agent(config)
        session = create_session(
            pool_size=max_workers,
            scraper_config=config,
            rate_limit_profile="interactive",
        )
        orchestrator = _build_ingest_orchestrator()

        _emit(event_callback, "status", {"phase": "feed", "message": "Fetching RSS feed..."})
        feed_urls = _collect_feed_urls(app, scraper_config)
        rolling_window_days = max(0, int(scraper_config.get("rolling_window_days", 0)))
        ingest_config = config.get("ingest") or {}
        if rolling_window_days > 0:
            _emit(
                event_callback,
                "status",
                {
                    "phase": "rolling_window",
                    "message": f"Loading papers from the past {rolling_window_days} days...",
                },
            )

        entries = _candidate_entries(
            orchestrator.fetch(
                mode=IngestMode.DAILY_WATCH,
                session=session,
                feed_urls=feed_urls,
                rolling_window_days=rolling_window_days,
                backend_names=ingest_config.get("backends"),
                user_agent=user_agent,
            )
        )

        total_entries = len(entries)
        _emit(event_callback, "feed", {"total": total_entries})

        entries, pre_filtered = _filter_existing_entries(app, entries)

        llm_client, interests_text = _create_llm_client(app)
        interest_profile = build_interest_profile(app)

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

        results = _collect_matched_results(
            entries,
            whitelists,
            scraper_config,
            session,
            llm_client,
            interests_text,
            config,
            total_entries=total_entries,
            event_callback=event_callback,
            interest_profile=interest_profile,
        )

        summary = _finalize_results(
            app,
            results,
            session,
            config,
            pre_filtered=pre_filtered,
            total_entries=total_entries,
            event_callback=event_callback,
            now=now,
        )
        _emit(event_callback, "done", summary)
        _finish_scrape_run(app, scrape_run_id, status="success")

        LOGGER.info(
            "Scrape complete: %s new, %s duplicates, %s matched out of %s entries",
            summary["new_papers"],
            summary["duplicates_skipped"],
            summary["total_matched"],
            summary["total_in_feed"],
        )
        return summary
    except Exception:
        _finish_scrape_run(app, scrape_run_id, status="error")
        raise
    finally:
        if session is not None:
            session.close()


def run_scrape(app) -> dict:
    return execute_scrape(app, event_callback=None)


def stream_or_start_scrape(app, force: bool = False):
    """Compatibility wrapper implemented in job manager module."""
    from app.services.jobs import SCRAPE_JOB_MANAGER

    return SCRAPE_JOB_MANAGER.stream_for_request(app, force=force)


def execute_historical_scrape(app, categories: list[str], start_dt: date, end_dt: date) -> dict:
    config = app.config["SCRAPER_CONFIG"]
    whitelists = config["whitelists"]
    scraper_config = config["scraper"]
    ingest_config = config.get("ingest") or {}
    max_workers = max(1, int(scraper_config.get("max_workers", DEFAULT_MAX_WORKERS)))
    user_agent = resolve_user_agent(config)
    session = create_session(
        pool_size=max_workers,
        scraper_config=config,
        rate_limit_profile="bulk",
    )
    try:
        orchestrator = _build_ingest_orchestrator()

        entries = _candidate_entries(
            orchestrator.fetch(
                mode=IngestMode.BACKFILL,
                session=session,
                categories=categories,
                start_dt=start_dt,
                end_dt=end_dt,
                max_results=2000,
                backend_names=ingest_config.get("backends"),
                user_agent=user_agent,
            )
        )
        total_entries = len(entries)
        if not entries:
            return _build_summary(0, 0, 0, 0)

        entries, pre_filtered = _filter_existing_entries(app, entries)

        llm_client, interests_text = _create_llm_client(app)
        interest_profile = build_interest_profile(app)
        enrich_entries_with_api_metadata(entries, session=session)

        results = _collect_matched_results(
            entries,
            whitelists,
            scraper_config,
            session,
            llm_client,
            interests_text,
            config,
            total_entries=total_entries,
            interest_profile=interest_profile,
        )

        return _finalize_results(
            app,
            results,
            session,
            config,
            pre_filtered=pre_filtered,
            total_entries=total_entries,
        )
    finally:
        session.close()
