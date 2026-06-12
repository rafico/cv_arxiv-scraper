"""Feature extraction stage of the ranking pipeline."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Protocol

from app.services.pipeline.candidate_generation import ScoredCandidate
from app.services.ranking import (
    MATCH_TYPE_WEIGHTS,
    RESOURCE_SIGNAL_WEIGHT,
    TERM_MATCH_WEIGHT,
    recency_multiplier,
    resolve_ranking_preferences,
)
from app.services.venues import parse_venue, venue_bonus

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FeatureVector:
    """All scoring features extracted from a candidate paper."""

    author_match_score: float = 0.0
    affiliation_match_score: float = 0.0
    title_match_score: float = 0.0
    term_count: int = 0
    term_score: float = 0.0
    resource_count: int = 0
    resource_score: float = 0.0
    recency: float = 1.0
    llm_relevance: float | None = None
    llm_bonus: float = 0.0
    citation_count: int | None = None
    citation_bonus: float = 0.0
    venue: str | None = None
    venue_year: int | None = None
    acceptance_status: str | None = None
    venue_bonus: float = 0.0
    interest_similarity: float | None = None
    interest_bonus: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "author_match_score": self.author_match_score,
            "affiliation_match_score": self.affiliation_match_score,
            "title_match_score": self.title_match_score,
            "term_count": self.term_count,
            "term_score": self.term_score,
            "resource_count": self.resource_count,
            "resource_score": self.resource_score,
            "recency": self.recency,
            "llm_relevance": self.llm_relevance,
            "llm_bonus": self.llm_bonus,
            "citation_count": self.citation_count,
            "citation_bonus": self.citation_bonus,
            "venue": self.venue,
            "venue_year": self.venue_year,
            "acceptance_status": self.acceptance_status,
            "venue_bonus": self.venue_bonus,
            "interest_similarity": self.interest_similarity,
            "interest_bonus": self.interest_bonus,
        }


class FeatureExtractor(Protocol):
    """Protocol for extracting scoring features from a candidate."""

    def extract(self, candidate: ScoredCandidate) -> FeatureVector: ...


class DefaultFeatureExtractor:
    """Extracts scoring features replicating compute_paper_score() logic."""

    def __init__(self, config: dict | None = None, interest_profile=None) -> None:
        self.preferences = resolve_ranking_preferences(config)
        self.interest_profile = interest_profile

    def _interest_features(self, entry: dict) -> tuple[float | None, float]:
        """Embed the candidate on the fly and score it against the interest profile.

        The vector is stashed in the entry (transient, like pdf_content) so
        _generate_embeddings can reuse it instead of encoding twice.
        """
        if self.interest_profile is None:
            return None, 0.0
        try:
            vector = entry.get("_embedding")
            if vector is None:
                from app.services.embeddings import get_embedding_service

                text = f"{entry.get('title', '')} {entry.get('abstract', '')}"
                vector = get_embedding_service().encode([text])[0]
                entry["_embedding"] = vector

            from app.services.interest_model import score_vector

            similarity = round(score_vector(self.interest_profile, vector), 4)
            return similarity, similarity * self.preferences["interest_weight"]
        except Exception:
            LOGGER.warning("Interest similarity scoring failed (non-fatal)", exc_info=True)
            return None, 0.0

    def extract(self, candidate: ScoredCandidate) -> FeatureVector:
        entry = candidate.entry_data
        match_types = candidate.match_types

        author_score = self.preferences.get("Author", MATCH_TYPE_WEIGHTS["Author"]) if "Author" in match_types else 0.0
        affiliation_score = (
            self.preferences.get("Affiliation", MATCH_TYPE_WEIGHTS["Affiliation"])
            if "Affiliation" in match_types
            else 0.0
        )
        title_score = self.preferences.get("Title", MATCH_TYPE_WEIGHTS["Title"]) if "Title" in match_types else 0.0

        term_count = len(candidate.matched_terms)
        term_score = term_count * TERM_MATCH_WEIGHT

        resource_count = len(entry.get("resource_links", []))
        resource_score = min(resource_count, 4) * RESOURCE_SIGNAL_WEIGHT

        llm_relevance = entry.get("llm_relevance_score")
        llm_bonus = (llm_relevance / 10.0) * self.preferences["ai_weight"] if llm_relevance is not None else 0.0

        citation_count = entry.get("citation_count")
        citation_bonus = 0.0
        if citation_count and citation_count > 0:
            citation_bonus = math.log1p(citation_count) * self.preferences["citation_weight"]

        venue_match = parse_venue(entry.get("comment", ""))
        venue_score = venue_bonus(
            venue_match.status if venue_match else None,
            self.preferences["venue_weight"],
        )

        interest_similarity, interest_bonus = self._interest_features(entry)

        recency = recency_multiplier(
            entry.get("publication_dt"),
            half_life_days=self.preferences["half_life_days"],
        )

        return FeatureVector(
            author_match_score=author_score,
            affiliation_match_score=affiliation_score,
            title_match_score=title_score,
            term_count=term_count,
            term_score=term_score,
            resource_count=resource_count,
            resource_score=resource_score,
            recency=recency,
            llm_relevance=llm_relevance,
            llm_bonus=llm_bonus,
            citation_count=citation_count,
            citation_bonus=citation_bonus,
            venue=venue_match.venue if venue_match else None,
            venue_year=venue_match.year if venue_match else None,
            acceptance_status=venue_match.status if venue_match else None,
            venue_bonus=venue_score,
            interest_similarity=interest_similarity,
            interest_bonus=interest_bonus,
        )
