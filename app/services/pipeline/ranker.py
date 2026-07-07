"""Ranking stage of the ranking pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from app.services.matching import MATCH_PRIORITY
from app.services.pipeline.candidate_generation import ScoredCandidate
from app.services.pipeline.features import DefaultFeatureExtractor, FeatureVector
from app.services.ranking import compute_paper_score


@dataclass(slots=True)
class RankedPaper:
    """A fully ranked paper with score breakdown."""

    entry_data: dict[str, Any]
    match_types: list[str]
    matched_terms: list[str]
    score: float
    features: FeatureVector
    pdf_content: bytes | None = None

    @property
    def match_type(self) -> str:
        return " + ".join(self.match_types)

    @property
    def match_priority(self) -> int:
        # Match types outside the whitelist trio (e.g. "Interest" dense-retrieval
        # candidates) sort after Title but before "no match at all".
        return min(
            (MATCH_PRIORITY.get(mt, 4) for mt in self.match_types),
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
            "llm_insights": entry.get("llm_insights", {}),
            "arxiv_comment": entry.get("comment") or None,
            "venue": self.features.venue,
            "venue_year": self.features.venue_year,
            "acceptance_status": self.features.acceptance_status,
            "interest_similarity": self.features.interest_similarity,
            # Transient on-the-fly SPECTER2 vector, reused by _generate_embeddings.
            "embedding": entry.get("_embedding"),
            "publication_dt": entry.get("publication_dt"),
            "publication_date": entry.get("publication_date", "Date Unknown"),
            # INVARIANT: pdf_content (PDF bytes, fetched once during candidate
            # generation) rides along in the result dict and is consumed by the
            # LAST pipeline steps — _generate_thumbnails AND _extract_sections.
            # Don't .pop() it early (use .get()), or section extraction silently
            # gets nothing. It is not persisted: _save_results maps explicit cols.
            "pdf_content": self.pdf_content,
        }


class WeightedSumRanker:
    """Ranks candidates using a weighted sum of features.

    Delegates scoring to ranking.compute_paper_score() to keep the formula
    in a single canonical location.
    """

    def __init__(self, config: dict | None = None, interest_profile=None) -> None:
        self.config = config
        self.extractor = DefaultFeatureExtractor(config, interest_profile=interest_profile)

    def rank(self, candidates: list[ScoredCandidate]) -> list[RankedPaper]:
        ranked = []
        for candidate in candidates:
            features = self.extractor.extract(candidate)
            score = compute_paper_score(
                match_types=candidate.match_types,
                matched_terms_count=len(candidate.matched_terms),
                publication_dt=candidate.entry_data.get("publication_dt"),
                resource_count=features.resource_count,
                llm_relevance_score=features.llm_relevance,
                citation_count=features.citation_count,
                acceptance_status=features.acceptance_status,
                interest_similarity=features.interest_similarity,
                config=self.config,
            )
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
