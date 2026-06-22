"""Tests verifying the ranking pipeline produces identical scores to the old code."""

from __future__ import annotations

from datetime import date

from app.services.pipeline import (
    FeatureVector,
    ScoredCandidate,
    WeightedSumRanker,
    WhitelistCandidateGenerator,
)
from app.services.ranking import compute_paper_score


class TestCheckAffiliations:
    """_check_affiliations reads prefetched data only — no network, no native code."""

    def _generator(self):
        return WhitelistCandidateGenerator(
            whitelists={"authors": [], "titles": [], "affiliations": ["MIT"]},
            scraper_config={},
        )

    def test_matches_from_stashed_pdf_text(self):
        entry = {
            "link": "https://arxiv.org/abs/1",
            "api_affiliations": "",
            "pdf_affiliation_text": "MIT CSAIL",
            "pdf_content": b"%PDF",
        }
        matches, pdf_content = self._generator()._check_affiliations(entry)
        assert matches == ["MIT"]
        assert pdf_content == b"%PDF"  # bytes flow through for thumbnails/sections

    def test_api_affiliation_match_short_circuits(self):
        entry = {"link": "https://arxiv.org/abs/2", "api_affiliations": "MIT", "pdf_content": None}
        matches, pdf_content = self._generator()._check_affiliations(entry)
        assert matches == ["MIT"]
        assert pdf_content is None

    def test_no_match_returns_empty(self):
        entry = {"link": "https://arxiv.org/abs/3", "api_affiliations": "", "pdf_affiliation_text": "Acme Corp"}
        matches, pdf_content = self._generator()._check_affiliations(entry)
        assert matches == []
        assert pdf_content is None


def _make_candidate(
    match_types: list[str],
    matched_terms: list[str],
    publication_dt: date | None = None,
    resource_links: list[dict] | None = None,
    llm_relevance_score: float | None = None,
    citation_count: int | None = None,
    comment: str = "",
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
        "comment": comment,
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


class TestVenueFeatures:
    def test_accepted_venue_flows_into_features_and_result(self):
        candidate = _make_candidate(["Title"], ["Vision"], date(2026, 4, 1), comment="Accepted to CVPR 2026 (oral)")
        ranked = WeightedSumRanker().rank([candidate])[0]

        assert ranked.features.venue == "CVPR"
        assert ranked.features.venue_year == 2026
        assert ranked.features.acceptance_status == "oral"
        assert ranked.features.venue_bonus == 12.0

        result = ranked.to_result_dict()
        assert result["venue"] == "CVPR"
        assert result["venue_year"] == 2026
        assert result["acceptance_status"] == "oral"
        assert result["arxiv_comment"] == "Accepted to CVPR 2026 (oral)"
        assert ranked.score == compute_paper_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 1),
            resource_count=0,
            acceptance_status="oral",
        )

    def test_venue_explanation_line(self):
        candidate = _make_candidate(["Title"], ["Vision"], date(2026, 4, 1), comment="Accepted to ECCV 2026")
        ranker = WeightedSumRanker()
        ranked = ranker.rank([candidate])[0]
        assert "Accepted at ECCV 2026" in ranker.generate_explanation(ranked)

    def test_no_comment_leaves_venue_unset(self):
        candidate = _make_candidate(["Title"], ["Vision"], date(2026, 4, 1))
        ranked = WeightedSumRanker().rank([candidate])[0]
        assert ranked.features.venue is None
        assert ranked.features.venue_bonus == 0.0
        assert ranked.to_result_dict()["arxiv_comment"] is None


class TestInterestFeatures:
    def _profile_and_service(self):
        from unittest.mock import MagicMock

        import numpy as np

        from app.services.interest_model import InterestProfile

        centroid = np.zeros(768, dtype=np.float32)
        centroid[0] = 1.0
        profile = InterestProfile(pos_centroid=centroid, neg_centroid=None, fingerprint=(5, 5))

        service = MagicMock()
        service.encode.return_value = np.asarray([centroid], dtype=np.float32)
        return profile, service

    def test_extractor_scores_against_profile_and_stashes_embedding(self):
        from unittest.mock import patch

        profile, service = self._profile_and_service()
        candidate = _make_candidate(["Title"], ["Vision"], date(2026, 4, 1))

        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            ranked = WeightedSumRanker(interest_profile=profile).rank([candidate])[0]

        assert ranked.features.interest_similarity == 1.0
        assert ranked.features.interest_bonus == 12.0
        assert candidate.entry_data["_embedding"] is not None

        result = ranked.to_result_dict()
        assert result["interest_similarity"] == 1.0
        assert result["embedding"] is not None
        assert ranked.score == compute_paper_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 1),
            resource_count=0,
            interest_similarity=1.0,
        )

    def test_no_profile_keeps_feature_inert(self):
        candidate = _make_candidate(["Title"], ["Vision"], date(2026, 4, 1))
        ranked = WeightedSumRanker().rank([candidate])[0]

        assert ranked.features.interest_similarity is None
        assert ranked.features.interest_bonus == 0.0
        assert ranked.to_result_dict()["embedding"] is None

    def test_encode_failure_degrades_to_inert(self):
        from unittest.mock import MagicMock, patch

        profile, _ = self._profile_and_service()
        service = MagicMock()
        service.encode.side_effect = RuntimeError("model unavailable")
        candidate = _make_candidate(["Title"], ["Vision"], date(2026, 4, 1))

        with patch("app.services.embeddings.get_embedding_service", return_value=service):
            ranked = WeightedSumRanker(interest_profile=profile).rank([candidate])[0]

        assert ranked.features.interest_similarity is None
        assert ranked.features.interest_bonus == 0.0


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
