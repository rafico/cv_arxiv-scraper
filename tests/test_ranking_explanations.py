"""Tests for ranking explanation generation."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

from app.services.ranking import generate_ranking_explanation, top_score_contributors


def _make_paper(**kwargs):
    """Create a mock paper with sensible defaults."""
    paper = MagicMock()
    paper.id = kwargs.get("id", 1)
    paper.match_type = kwargs.get("match_type", "Author")
    paper.matched_terms_list = kwargs.get("matched_terms_list", ["Kaiming He"])
    paper.citation_count = kwargs.get("citation_count")
    paper.publication_dt = kwargs.get("publication_dt", date.today())
    paper.resource_links_list = kwargs.get("resource_links_list", [])
    paper.llm_relevance_score = kwargs.get("llm_relevance_score")
    paper.abstract_text = kwargs.get("abstract_text", "A paper about neural networks.")
    paper.title = kwargs.get("title", "Test Paper Title")
    paper.venue = kwargs.get("venue")
    paper.venue_year = kwargs.get("venue_year")
    paper.acceptance_status = kwargs.get("acceptance_status")
    paper.interest_similarity = kwargs.get("interest_similarity")
    return paper


class TestGenerateRankingExplanation:
    def test_author_match(self):
        paper = _make_paper(match_type="Author", matched_terms_list=["Kaiming He"])
        explanations = generate_ranking_explanation(paper)
        assert any("Kaiming He" in e for e in explanations)

    def test_affiliation_match(self):
        paper = _make_paper(match_type="Affiliation", matched_terms_list=["Stanford"])
        explanations = generate_ranking_explanation(paper)
        assert any("Stanford" in e for e in explanations)

    def test_title_match(self):
        paper = _make_paper(match_type="Title", matched_terms_list=["NeRF", "3D"])
        explanations = generate_ranking_explanation(paper)
        assert any("NeRF" in e for e in explanations)

    def test_multi_match(self):
        paper = _make_paper(match_type="Author+Affiliation", matched_terms_list=["He", "MIT"])
        explanations = generate_ranking_explanation(paper)
        assert len(explanations) >= 2

    def test_high_citations(self):
        paper = _make_paper(citation_count=100)
        explanations = generate_ranking_explanation(paper)
        assert any("100" in e and "cited" in e.lower() for e in explanations)

    def test_low_citations_not_mentioned(self):
        paper = _make_paper(citation_count=5)
        explanations = generate_ranking_explanation(paper)
        assert not any("cited" in e.lower() for e in explanations)

    def test_recent_paper(self):
        paper = _make_paper(publication_dt=date.today())
        explanations = generate_ranking_explanation(paper)
        assert any("recently" in e.lower() for e in explanations)

    def test_venue_acceptance_explained(self):
        paper = _make_paper(venue="CVPR", venue_year=2026, acceptance_status="oral")
        explanations = generate_ranking_explanation(paper)
        assert any("Accepted at CVPR 2026 (oral)" in e for e in explanations)

    def test_mentioned_venue_not_explained(self):
        paper = _make_paper(venue="ICLR", acceptance_status="mentioned")
        explanations = generate_ranking_explanation(paper)
        assert not any("ICLR" in e for e in explanations)

    def test_high_interest_similarity_explained(self):
        paper = _make_paper(interest_similarity=0.8)
        explanations = generate_ranking_explanation(paper)
        assert any("matches papers you saved" in e.lower() for e in explanations)

    def test_low_interest_similarity_not_explained(self):
        paper = _make_paper(interest_similarity=0.1)
        explanations = generate_ranking_explanation(paper)
        assert not any("matches papers you saved" in e.lower() for e in explanations)

    def test_old_paper_no_recency(self):
        paper = _make_paper(publication_dt=date.today() - timedelta(days=60))
        explanations = generate_ranking_explanation(paper)
        assert not any("recently" in e.lower() for e in explanations)

    def test_ai_relevance(self):
        paper = _make_paper(llm_relevance_score=8.5)
        explanations = generate_ranking_explanation(paper)
        assert any("AI rated" in e for e in explanations)

    def test_resources_available(self):
        paper = _make_paper(resource_links_list=[{"url": "https://github.com/test", "label": "code"}])
        explanations = generate_ranking_explanation(paper)
        assert any("code" in e.lower() or "dataset" in e.lower() for e in explanations)

    def test_no_match_type(self):
        paper = _make_paper(match_type="", matched_terms_list=[])
        explanations = generate_ranking_explanation(paper)
        # Should still work, just no match explanations
        assert isinstance(explanations, list)


class TestTopScoreContributors:
    def test_returns_top_three_sorted_descending(self):
        breakdown = {
            "match_score": 44.0,
            "term_score": 6.0,
            "interest_bonus": 12.0,
            "venue_bonus": 8.0,
            "ai_bonus": 2.0,
        }
        top = top_score_contributors(breakdown)
        assert [c["label"] for c in top] == ["Match", "For you", "Venue"]
        assert [c["value"] for c in top] == [44.0, 12.0, 8.0]

    def test_excludes_zero_and_negative_factors(self):
        breakdown = {"match_score": 10.0, "term_score": 0.0, "interest_bonus": -3.0}
        top = top_score_contributors(breakdown)
        assert [c["label"] for c in top] == ["Match"]

    def test_empty_or_none_breakdown_returns_empty(self):
        assert top_score_contributors(None) == []
        assert top_score_contributors({}) == []
        assert top_score_contributors({"match_score": 0.0, "recency_multiplier": 0.9}) == []

    def test_pct_is_normalised_to_strongest(self):
        breakdown = {"match_score": 40.0, "interest_bonus": 20.0, "term_score": 10.0}
        top = top_score_contributors(breakdown)
        assert top[0]["pct"] == 100
        assert top[1]["pct"] == 50
        assert top[2]["pct"] == 25

    def test_limit_is_respected(self):
        breakdown = {"match_score": 5.0, "interest_bonus": 4.0, "term_score": 3.0, "venue_bonus": 2.0}
        assert len(top_score_contributors(breakdown, limit=2)) == 2

    def test_each_factor_has_a_colour_token(self):
        top = top_score_contributors({"match_score": 10.0, "interest_bonus": 5.0})
        assert all(c["color"] in {"accent", "info", "priority", "save"} for c in top)
