"""Candidate generation stage of the ranking pipeline."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import requests

from app.enums import MatchType
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


class WhitelistCandidateGenerator:
    """Generates candidates by matching against author/title/affiliation whitelists.

    Entries with no whitelist hit get a second chance through the dense-retrieval
    interest gate: when the learned ranker (or the centroid interest profile as
    cold-start fallback) scores their embedding above a configurable threshold,
    they are admitted with match type "Interest" and their score in
    ``raw_features``. The per-scrape ``candidate_top_k`` cap is enforced by the
    caller (scrape_engine keeps the top-K *by score* across the whole run, not
    the first K seen in stream order), so a bad model can never flood the feed.
    """

    def __init__(
        self,
        whitelists: dict[str, list],
        scraper_config: dict[str, Any],
        muted: dict[str, list] | None = None,
        session: requests.Session | None = None,
        interest_scorer=None,
        interest_settings: dict | None = None,
    ) -> None:
        self.whitelists = whitelists
        self.scraper_config = scraper_config
        self.muted = muted or {"authors": [], "affiliations": [], "topics": []}
        self.session = session
        # Interest gate state. The scorer maps an embedding vector to a
        # probability-like score in [0, 1] (or None). It is resolved lazily on
        # first use — scrape_engine constructs this generator without config
        # access, so settings come from the learned-ranker runtime snapshot
        # (refreshed by DefaultFeatureExtractor construction at scrape start).
        self._interest_lock = threading.Lock()
        self._interest_scorer = interest_scorer
        self._interest_settings = interest_settings
        self._interest_resolved = interest_scorer is not None

    def generate(self, papers: list[dict[str, Any]]) -> list[ScoredCandidate]:
        candidates = []
        for entry_data in papers:
            candidate = self.process_single(entry_data)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _check_fast_matches(self, entry_data: dict) -> dict[str, list[str]]:
        """Check title and author matches -- no network needed."""
        return {
            "Author": check_author_match(entry_data["authors_list"], self.whitelists["authors"]),
            "Title": check_whitelist_match(
                [entry_data["title"], entry_data.get("abstract", "")],
                self.whitelists["titles"],
            ),
        }

    def _check_affiliations(self, entry_data: dict) -> tuple[list[str], bytes | None]:
        """Check affiliation matches from API metadata and prefetched PDF header text.

        The PDF download and native pdfplumber parse happen once per scrape, isolated, in
        ``_prefetch_affiliation_text`` — this per-paper worker only reads the stashed
        ``pdf_affiliation_text`` / ``pdf_content``, doing no network or native work.
        """
        affiliation_matches: list[str] = []

        api_affiliations = entry_data.get("api_affiliations", "")
        if api_affiliations:
            affiliation_matches = check_whitelist_match([api_affiliations], self.whitelists["affiliations"])

        if not affiliation_matches:
            affiliation_text = entry_data.get("pdf_affiliation_text", "")
            if affiliation_text:
                affiliation_matches = check_whitelist_match([affiliation_text], self.whitelists["affiliations"])

        return affiliation_matches, entry_data.get("pdf_content")

    def _is_muted(self, entry_data: dict) -> bool:
        """Check if a paper should be suppressed by mute filters."""
        from app.services.summary import extract_topic_tags

        if check_author_match(entry_data["authors_list"], self.muted["authors"]):
            return True
        if check_whitelist_match([entry_data.get("api_affiliations", "")], self.muted["affiliations"]):
            return True
        topic_tags = extract_topic_tags(entry_data["title"], entry_data.get("abstract", ""))
        if check_whitelist_match(topic_tags, self.muted["topics"]):
            return True
        return False

    def _resolve_interest_gate(self) -> None:
        """Build the interest scorer once per generator (thread-safe, non-fatal)."""
        with self._interest_lock:
            if self._interest_resolved:
                return
            self._interest_resolved = True
            try:
                from app.services import learned_ranker
                from app.services.interest_model import get_cached_interest_profile

                prefs = learned_ranker.get_runtime_learned_prefs()
                if self._interest_settings is None:
                    self._interest_settings = prefs
                settings = self._interest_settings
                if not settings.get("enabled", True) or int(settings.get("candidate_top_k", 0)) <= 0:
                    return

                model = learned_ranker.peek_learned_model()
                profile = get_cached_interest_profile()
                description_vector = None
                try:
                    ref = learned_ranker.get_runtime_active_profile()
                    description_vector = learned_ranker._description_vector(ref)
                except Exception:  # pragma: no cover - description blend is best-effort
                    description_vector = None
                if model is None and profile is None and description_vector is None:
                    return
                blend = float(settings.get("blend", 0.7))

                def scorer(vector) -> float | None:
                    signal, _source = learned_ranker.interest_signal(vector, profile, model, blend, description_vector)
                    if signal is None:
                        return None
                    # Map the [-1, 1] signal to [0, 1] so the threshold reads as
                    # a probability (a pure-LR signal maps back to its probability).
                    return (signal + 1.0) / 2.0

                self._interest_scorer = scorer
            except Exception:
                LOGGER.warning("Interest candidate gate unavailable (non-fatal)", exc_info=True)

    def _interest_candidate(self, entry_data: dict) -> ScoredCandidate | None:
        """Admit a non-whitelist entry when the interest model scores it highly."""
        self._resolve_interest_gate()
        if self._interest_scorer is None:
            return None
        settings = self._interest_settings or {}
        threshold = float(settings.get("candidate_threshold", 0.6))

        if self._is_muted(entry_data):
            return None

        try:
            vector = entry_data.get("_embedding")
            if vector is None:
                from app.services.embeddings import get_embedding_service

                text = f"{entry_data.get('title', '')} {entry_data.get('abstract', '')}"
                vector = get_embedding_service().encode([text])[0]
                # Stash for reuse by feature extraction / _generate_embeddings.
                entry_data["_embedding"] = vector
            score = self._interest_scorer(vector)
        except Exception:
            LOGGER.warning("Interest candidate scoring failed (non-fatal)", exc_info=True)
            return None

        if score is None or score < threshold:
            return None

        return ScoredCandidate(
            entry_data=entry_data,
            match_types=[MatchType.INTEREST.value],
            matched_terms=[],
            pdf_content=entry_data.get("pdf_content"),
            raw_features={"interest_candidate_score": round(float(score), 4)},
        )

    def process_single(self, entry_data: dict) -> ScoredCandidate | None:
        """Process a single paper entry through the candidate generation pipeline."""
        fast_matches = self._check_fast_matches(entry_data)
        affiliation_matches, pdf_content = self._check_affiliations(entry_data)

        category_matches = {**fast_matches, "Affiliation": affiliation_matches}

        if not any(category_matches.values()):
            # No whitelist hit: dense-retrieval second chance via the learned
            # interest model (bounded + thresholded; None when unavailable).
            return self._interest_candidate(entry_data)

        if self._is_muted(entry_data):
            return None

        match_types = [name for name, terms in category_matches.items() if terms]
        matched_terms = dedupe_preserve_order(term for terms in category_matches.values() for term in terms)

        return ScoredCandidate(
            entry_data=entry_data,
            match_types=match_types,
            matched_terms=matched_terms,
            pdf_content=pdf_content,
        )
