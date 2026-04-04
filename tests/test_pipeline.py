"""Tests verifying the ranking pipeline produces identical scores to the old code."""

from __future__ import annotations

from datetime import date

from app.services.pipeline import (
    FeatureVector,
    ScoredCandidate,
    WeightedSumRanker,
)
from app.services.ranking import compute_paper_score


def _make_candidate(
    match_types: list[str],
    matched_terms: list[str],
    publication_dt: date | None = None,
    resource_links: list[dict] | None = None,
    llm_relevance_score: float | None = None,
    citation_count: int | None = None,
) -> ScoredCandidate:
    entry = {
        "arxiv_id": "2401.00001",
        "link": "https://arxiv.org/abs/2401.00001",
        "title": "Test Paper",
        "author": "Author A",
        "authors_list": ["Author A"],
        "abstract": "An abstract.",
        "publication_dt": publication_dt,
        "publication_date": publication_dt.isoformat() if publication_dt else "Date Unknown",
        "resource_links": resource_links or [],
        "categories": [],
        "llm_relevance_score": llm_relevance_score,
        "citation_count": citation_count,
    }
    return ScoredCandidate(
        entry_data=entry,
        match_types=match_types,
        matched_terms=matched_terms,
    )


class TestPipelineScoreParity:
    """Verify the pipeline produces identical scores to compute_paper_score()."""

    def test_author_match_only(self):
        candidate = _make_candidate(["Author"], ["Andrew Y. Ng"], date(2026, 4, 1))
        ranker = WeightedSumRanker()
        ranked = ranker.rank([candidate])
        old_score = compute_paper_score(
            match_types=["Author"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 1),
            resource_count=0,
        )
        assert ranked[0].score == old_score

    def test_multiple_match_types(self):
        candidate = _make_candidate(
            ["Author", "Title"],
            ["Andrew Y. Ng", "Few Shot"],
            date(2026, 4, 1),
        )
        ranker = WeightedSumRanker()
        ranked = ranker.rank([candidate])
        old_score = compute_paper_score(
            match_types=["Author", "Title"],
            matched_terms_count=2,
            publication_dt=date(2026, 4, 1),
            resource_count=0,
        )
        assert ranked[0].score == old_score

    def test_with_resources(self):
        candidate = _make_candidate(
            ["Affiliation"],
            ["Google"],
            date(2026, 4, 1),
            resource_links=[{"url": "https://github.com/example", "type": "code"}] * 3,
        )
        ranker = WeightedSumRanker()
        ranked = ranker.rank([candidate])
        old_score = compute_paper_score(
            match_types=["Affiliation"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 1),
            resource_count=3,
        )
        assert ranked[0].score == old_score

    def test_with_llm_relevance(self):
        candidate = _make_candidate(
            ["Title"],
            ["Zero Shot"],
            date(2026, 4, 1),
            llm_relevance_score=8.5,
        )
        ranker = WeightedSumRanker()
        ranked = ranker.rank([candidate])
        old_score = compute_paper_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 1),
            resource_count=0,
            llm_relevance_score=8.5,
        )
        assert ranked[0].score == old_score

    def test_with_citations(self):
        candidate = _make_candidate(
            ["Author"],
            ["Yoshua Bengio"],
            date(2026, 4, 1),
            citation_count=150,
        )
        ranker = WeightedSumRanker()
        ranked = ranker.rank([candidate])
        old_score = compute_paper_score(
            match_types=["Author"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 1),
            resource_count=0,
            citation_count=150,
        )
        assert ranked[0].score == old_score

    def test_no_publication_date(self):
        candidate = _make_candidate(["Title"], ["Nerf"])
        ranker = WeightedSumRanker()
        ranked = ranker.rank([candidate])
        old_score = compute_paper_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=None,
            resource_count=0,
        )
        assert ranked[0].score == old_score

    def test_all_features_combined(self):
        candidate = _make_candidate(
            ["Author", "Affiliation", "Title"],
            ["Andrew Y. Ng", "Stanford", "Remote Sensing"],
            date(2026, 3, 28),
            resource_links=[{"url": "x", "type": "code"}] * 2,
            llm_relevance_score=7.0,
            citation_count=50,
        )
        ranker = WeightedSumRanker()
        ranked = ranker.rank([candidate])
        old_score = compute_paper_score(
            match_types=["Author", "Affiliation", "Title"],
            matched_terms_count=3,
            publication_dt=date(2026, 3, 28),
            resource_count=2,
            llm_relevance_score=7.0,
            citation_count=50,
        )
        assert ranked[0].score == old_score


class TestFeatureVector:
    def test_to_dict(self):
        fv = FeatureVector(author_match_score=44.0, term_count=2, term_score=6.0)
        d = fv.to_dict()
        assert d["author_match_score"] == 44.0
        assert d["term_count"] == 2
        assert d["term_score"] == 6.0


class TestRankerSorting:
    def test_sorts_by_score_descending(self):
        c1 = _make_candidate(["Title"], ["Few Shot"], date(2026, 4, 1))
        c2 = _make_candidate(["Author"], ["Andrew Y. Ng"], date(2026, 4, 1))
        ranker = WeightedSumRanker()
        ranked = ranker.rank([c1, c2])
        assert ranked[0].score >= ranked[1].score

    def test_to_result_dict_backward_compatible(self):
        candidate = _make_candidate(["Author", "Title"], ["Ng", "Nerf"], date(2026, 4, 1))
        candidate.entry_data["summary_text"] = "A summary"
        candidate.entry_data["topic_tags"] = ["cv", "nerf"]
        ranker = WeightedSumRanker()
        ranked = ranker.rank([candidate])
        result = ranked[0].to_result_dict()

        assert "arxiv_id" in result
        assert "title" in result
        assert "authors" in result
        assert "link" in result
        assert "pdf_link" in result
        assert "match_type" in result
        assert "match_types" in result
        assert "matches" in result
        assert "paper_score" in result
        assert "publication_dt" in result
        assert result["match_type"] == "Author + Title"
        assert result["matches"] == ["Ng", "Nerf"]


class TestRankerExplanation:
    def test_author_explanation(self):
        candidate = _make_candidate(["Author"], ["Andrew Y. Ng"], date(2026, 4, 1))
        ranker = WeightedSumRanker()
        ranked = ranker.rank([candidate])
        explanations = ranker.generate_explanation(ranked[0])
        assert any("Matched author" in e for e in explanations)

    def test_citation_explanation(self):
        candidate = _make_candidate(["Title"], ["SAR"], date(2026, 4, 1), citation_count=100)
        ranker = WeightedSumRanker()
        ranked = ranker.rank([candidate])
        explanations = ranker.generate_explanation(ranked[0])
        assert any("cited" in e.lower() for e in explanations)

    def test_recency_explanation(self):
        candidate = _make_candidate(["Title"], ["MOT"], date.today())
        ranker = WeightedSumRanker()
        ranked = ranker.rank([candidate])
        explanations = ranker.generate_explanation(ranked[0])
        assert any("recently" in e.lower() for e in explanations)
