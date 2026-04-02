"""Candidate generation stage of the ranking pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

from app.services.http_client import request_with_backoff
from app.services.matching import (
    check_author_match,
    check_whitelist_match,
    dedupe_preserve_order,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ScoredCandidate:
    """A paper that passed candidate generation with its match metadata."""

    entry_data: dict[str, Any]
    match_types: list[str]
    matched_terms: list[str]
    pdf_content: bytes | None = None
    raw_features: dict[str, Any] = field(default_factory=dict)


class CandidateGenerator(Protocol):
    """Protocol for candidate generation strategies."""

    def generate(self, papers: list[dict[str, Any]]) -> list[ScoredCandidate]: ...


class WhitelistCandidateGenerator:
    """Generates candidates by matching against author/title/affiliation whitelists.

    Extracts logic from scrape_engine._check_fast_matches() and the affiliation
    matching portion of _process_paper_entry().
    """

    def __init__(
        self,
        whitelists: dict[str, list],
        scraper_config: dict[str, Any],
        muted: dict[str, list] | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.whitelists = whitelists
        self.scraper_config = scraper_config
        self.muted = muted or {"authors": [], "affiliations": [], "topics": []}
        self.session = session

    def generate(self, papers: list[dict[str, Any]]) -> list[ScoredCandidate]:
        candidates = []
        for entry_data in papers:
            candidate = self._process_single(entry_data)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _check_fast_matches(self, entry_data: dict) -> dict[str, list[str]]:
        """Check title and author matches -- no network needed."""
        return {
            "Author": check_author_match(
                entry_data["authors_list"], self.whitelists["authors"]
            ),
            "Title": check_whitelist_match(
                [entry_data["title"], entry_data.get("abstract", "")],
                self.whitelists["titles"],
            ),
        }

    def _check_affiliations(self, entry_data: dict) -> tuple[list[str], bytes | None]:
        """Check affiliation matches from API metadata and PDF headers."""
        affiliation_matches: list[str] = []
        pdf_content: bytes | None = None

        api_affiliations = entry_data.get("api_affiliations", "")
        if api_affiliations:
            affiliation_matches = check_whitelist_match(
                [api_affiliations], self.whitelists["affiliations"]
            )

        if not affiliation_matches:
            link = entry_data["link"]
            pdf_url = link.replace("/abs/", "/pdf/")
            try:
                from app.services.enrichment import extract_affiliation_text

                pdf_response = request_with_backoff(
                    "GET",
                    pdf_url,
                    timeout=30,
                    attempts=self.scraper_config.get("pdf_attempts", 2),
                    base_delay=1.0,
                    session=self.session,
                )
                pdf_content = pdf_response.content
                affiliation_text = extract_affiliation_text(
                    pdf_content,
                    lines_start=self.scraper_config.get("pdf_lines_start", 2),
                    max_header_lines=self.scraper_config.get(
                        "pdf_max_header_lines",
                        self.scraper_config.get("pdf_lines_end", 50),
                    ),
                    smart_header=self.scraper_config.get("pdf_smart_header", True),
                )
                if affiliation_text:
                    affiliation_matches = check_whitelist_match(
                        [affiliation_text], self.whitelists["affiliations"]
                    )
            except Exception as exc:
                LOGGER.warning(
                    "Error fetching PDF for %s: %s", entry_data.get("link"), exc
                )

        return affiliation_matches, pdf_content

    def _is_muted(self, entry_data: dict) -> bool:
        """Check if a paper should be suppressed by mute filters."""
        from app.services.summary import extract_topic_tags

        if check_author_match(entry_data["authors_list"], self.muted["authors"]):
            return True
        if check_whitelist_match(
            [entry_data.get("api_affiliations", "")], self.muted["affiliations"]
        ):
            return True
        topic_tags = extract_topic_tags(
            entry_data["title"], entry_data.get("abstract", "")
        )
        if check_whitelist_match(topic_tags, self.muted["topics"]):
            return True
        return False

    def _process_single(self, entry_data: dict) -> ScoredCandidate | None:
        """Process a single paper entry through the candidate generation pipeline."""
        fast_matches = self._check_fast_matches(entry_data)
        affiliation_matches, pdf_content = self._check_affiliations(entry_data)

        category_matches = {**fast_matches, "Affiliation": affiliation_matches}

        if not any(category_matches.values()):
            return None

        if self._is_muted(entry_data):
            return None

        match_types = [name for name, terms in category_matches.items() if terms]
        matched_terms = dedupe_preserve_order(
            term for terms in category_matches.values() for term in terms
        )

        return ScoredCandidate(
            entry_data=entry_data,
            match_types=match_types,
            matched_terms=matched_terms,
            pdf_content=pdf_content,
        )
