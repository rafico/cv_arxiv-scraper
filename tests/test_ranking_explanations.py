"""Tests for ranking explanation generation."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

from app.services.ranking import generate_ranking_explanation


def _make_paper(**kwargs):
    """Create a mock paper with sensible defaults."""
    paper = MagicMock()
    paper.id = kwargs.get("id", 1)
    paper.match_type = kwargs.get("match_type", "Author")
    paper.matched_terms_list = kwargs.get("matched_terms_list", ["Kaiming He"])
    paper.citation_count = kwargs.get("citation_count", None)
    paper.publication_dt = kwargs.get("publication_dt", date.today())
    paper.resource_links_list = kwargs.get("resource_links_list", [])
    paper.llm_relevance_score = kwargs.get("llm_relevance_score", None)
    paper.abstract_text = kwargs.get("abstract_text", "A paper about neural networks.")
    paper.title = kwargs.get("title", "Test Paper Title")
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
