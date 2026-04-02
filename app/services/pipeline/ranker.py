"""Ranking stage of the ranking pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from app.services.matching import MATCH_PRIORITY
from app.services.pipeline.candidate_generation import ScoredCandidate
from app.services.pipeline.features import (
    DefaultFeatureExtractor,
    FeatureExtractor,
    FeatureVector,
)


@dataclass(slots=True)
class RankedPaper:
    """A fully ranked paper with score breakdown."""

    entry_data: dict[str, Any]
    match_types: list[str]
    matched_terms: list[str]
    score: float
    features: FeatureVector
    pdf_content: bytes | None = None
    explanations: list[str] = field(default_factory=list)

    @property
    def match_type(self) -> str:
        return " + ".join(self.match_types)

    @property
    def match_priority(self) -> int:
        return min(
            (MATCH_PRIORITY[mt] for mt in self.match_types),
            default=999,
        )

    def to_result_dict(self) -> dict[str, Any]:
        """Convert to the legacy result dict format for _save_results() compatibility."""
        entry = self.entry_data
        return {
            "arxiv_id": entry.get("arxiv_id"),
            "title": entry.get("title", ""),
            "authors": entry.get("author", ""),
            "link": entry.get("link", ""),
            "pdf_link": entry.get("link", "").replace("/abs/", "/pdf/"),
            "abstract_text": entry.get("abstract", ""),
            "summary_text": entry.get("summary_text", ""),
            "topic_tags": entry.get("topic_tags", []),
            "categories": entry.get("categories", []),
            "resource_links": entry.get("resource_links", []),
            "matches": self.matched_terms,
            "match_types": self.match_types,
            "match_type": self.match_type,
            "match_priority": self.match_priority,
            "paper_score": self.score,
            "llm_relevance_score": self.features.llm_relevance,
            "publication_dt": entry.get("publication_dt"),
            "publication_date": entry.get("publication_date", "Date Unknown"),
            "pdf_content": self.pdf_content,
        }


class Ranker(Protocol):
    """Protocol for ranking strategies."""

    def rank(self, candidates: list[ScoredCandidate]) -> list[RankedPaper]: ...


class WeightedSumRanker:
    """Ranks candidates using a weighted sum of features.

    Replicates the exact scoring behavior of compute_paper_score().
    """

    def __init__(
        self,
        config: dict | None = None,
        feature_extractor: FeatureExtractor | None = None,
    ) -> None:
        self.config = config
        self.extractor = feature_extractor or DefaultFeatureExtractor(config)

    def rank(self, candidates: list[ScoredCandidate]) -> list[RankedPaper]:
        ranked = []
        for candidate in candidates:
            features = self.extractor.extract(candidate)
            score = self._compute_score(features)
            ranked.append(
                RankedPaper(
                    entry_data=candidate.entry_data,
                    match_types=candidate.match_types,
                    matched_terms=candidate.matched_terms,
                    score=score,
                    features=features,
                    pdf_content=candidate.pdf_content,
                )
            )

        ranked.sort(
            key=lambda r: (
                r.score,
                r.entry_data.get("publication_dt") or date.min,
            ),
            reverse=True,
        )
        return ranked

    def _compute_score(self, features: FeatureVector) -> float:
        match_score = (
            features.author_match_score
            + features.affiliation_match_score
            + features.title_match_score
        )
        raw = (
            match_score
            + features.term_score
            + features.resource_score
            + features.llm_bonus
            + features.citation_bonus
        )
        return round(raw * features.recency, 3)

    def generate_explanation(self, ranked_paper: RankedPaper) -> list[str]:
        """Generate human-readable explanation strings for a ranked paper."""
        explanations: list[str] = []
        features = ranked_paper.features
        matched_terms = ranked_paper.matched_terms[:3]

        for mt in ranked_paper.match_types:
            if mt == "Author":
                if matched_terms:
                    explanations.append(f"Matched author: {matched_terms[0]}")
                else:
                    explanations.append("Matched author in your watchlist")
            elif mt == "Affiliation":
                if matched_terms:
                    explanations.append(f"From tracked institution: {matched_terms[0]}")
                else:
                    explanations.append("From a tracked institution")
            elif mt == "Title":
                if matched_terms:
                    explanations.append(f"Title matches: {', '.join(matched_terms)}")
                else:
                    explanations.append("Title matches your interests")

        if features.citation_count and features.citation_count > 10:
            explanations.append(f"Highly cited ({features.citation_count} citations)")

        if features.recency > 0.9:
            explanations.append("Published very recently")

        if features.llm_relevance is not None and features.llm_relevance >= 7:
            explanations.append(
                f"AI rated highly relevant ({features.llm_relevance:.0f}/10)"
            )

        if ranked_paper.entry_data.get("resource_links"):
            explanations.append("Code or dataset available")

        return explanations
