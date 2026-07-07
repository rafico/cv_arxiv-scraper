"""Feature extraction stage of the ranking pipeline."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

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
    # "learned" (LR model), "centroid" (interest profile), or None (no signal).
    interest_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_UNRESOLVED = object()


class DefaultFeatureExtractor:
    """Extracts scoring features replicating compute_paper_score() logic."""

    def __init__(self, config: dict | None = None, interest_profile=None) -> None:
        self.preferences = resolve_ranking_preferences(config)
        self.interest_profile = interest_profile
        # Resolve the learned-ranker preferences up front: this also snapshots
        # them module-wide so the candidate-generation interest gate (which has
        # no access to the product config) sees the same settings this scrape.
        try:
            from app.services.learned_ranker import learned_preferences

            self.learned_preferences = learned_preferences(config)
        except Exception:  # pragma: no cover - preference resolution is best-effort
            LOGGER.warning("Learned preferences resolution failed (non-fatal)", exc_info=True)
            self.learned_preferences = {"enabled": False, "blend": 0.7}
        # Active profile's editable NL description, embedded once and blended into
        # the interest signal so a description-only cold-start profile still ranks.
        self.description_vector = None
        try:
            if self.learned_preferences.get("enabled", True):
                from app.services.learned_ranker import active_description_vector

                self.description_vector = active_description_vector()
        except Exception:  # pragma: no cover - description blend is best-effort
            LOGGER.warning("Profile description vector unavailable (non-fatal)", exc_info=True)
        self._learned_model = _UNRESOLVED

    def _resolve_learned_model(self):
        """Lazily peek the trained learned-ranker model (no DB, no training)."""
        if self._learned_model is _UNRESOLVED:
            model = None
            try:
                if self.learned_preferences.get("enabled", True):
                    from app.services.learned_ranker import peek_learned_model

                    model = peek_learned_model()
            except Exception:
                LOGGER.warning("Learned model lookup failed (non-fatal)", exc_info=True)
            self._learned_model = model
        return self._learned_model

    def _interest_features(self, entry: dict) -> tuple[float | None, float, str | None]:
        """Embed the candidate on the fly and score its interest signal.

        Uses the learned-ranker probability blended with the centroid profile
        when the trained model is available, else the centroid alone (existing
        behavior). The vector is stashed in the entry (transient, like
        pdf_content) so _generate_embeddings can reuse it instead of encoding
        twice.
        """
        model = self._resolve_learned_model()
        if self.interest_profile is None and model is None and self.description_vector is None:
            return None, 0.0, None
        try:
            vector = entry.get("_embedding")
            if vector is None:
                from app.services.embeddings import get_embedding_service

                text = f"{entry.get('title', '')} {entry.get('abstract', '')}"
                vector = get_embedding_service().encode([text])[0]
                entry["_embedding"] = vector

            from app.services.learned_ranker import interest_signal

            signal, source = interest_signal(
                vector,
                self.interest_profile,
                model,
                float(self.learned_preferences.get("blend", 0.7)),
                self.description_vector,
            )
            if signal is None:
                return None, 0.0, None
            similarity = round(signal, 4)
            return similarity, similarity * self.preferences["interest_weight"], source
        except Exception:
            LOGGER.warning("Interest similarity scoring failed (non-fatal)", exc_info=True)
            return None, 0.0, None

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

        interest_similarity, interest_bonus, interest_source = self._interest_features(entry)

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
            interest_source=interest_source,
        )
