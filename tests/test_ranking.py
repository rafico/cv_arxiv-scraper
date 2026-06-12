import unittest
from datetime import date, timedelta
from unittest.mock import patch

from app.services.ranking import (
    combined_rank_score,
    compute_feedback_delta,
    compute_paper_score,
    recency_multiplier,
)


class RankingTests(unittest.TestCase):
    def test_newer_papers_score_higher_with_same_signals(self):
        today = date.today()
        recent = compute_paper_score(
            match_types=["Author"],
            matched_terms_count=2,
            publication_dt=today,
            resource_count=1,
        )
        older = compute_paper_score(
            match_types=["Author"],
            matched_terms_count=2,
            publication_dt=today - timedelta(days=45),
            resource_count=1,
        )
        self.assertGreater(recent, older)

    def test_feedback_delta_weights(self):
        self.assertEqual(compute_feedback_delta("save"), 10)
        self.assertEqual(compute_feedback_delta("skip"), -9)
        self.assertEqual(compute_feedback_delta("unknown"), 0)

    def test_combined_rank_score(self):
        self.assertEqual(combined_rank_score(10.0, 4), 15.0)

    def test_llm_relevance_bonus_increases_score(self):
        today = date.today()
        without_llm = compute_paper_score(
            match_types=["Author"],
            matched_terms_count=2,
            publication_dt=today,
            resource_count=1,
            llm_relevance_score=None,
        )
        with_llm = compute_paper_score(
            match_types=["Author"],
            matched_terms_count=2,
            publication_dt=today,
            resource_count=1,
            llm_relevance_score=8.0,
        )
        self.assertGreater(with_llm, without_llm)

    @patch("app.services.ranking.utc_today", return_value=date(2026, 3, 20))
    def test_recency_multiplier_uses_utc_today(self, _mock_today):
        self.assertEqual(recency_multiplier(date(2026, 3, 20)), 1.0)

    def test_venue_acceptance_increases_score(self):
        today = date.today()
        base_kwargs = {
            "match_types": ["Title"],
            "matched_terms_count": 1,
            "publication_dt": today,
            "resource_count": 0,
        }
        no_venue = compute_paper_score(**base_kwargs)
        mentioned = compute_paper_score(**base_kwargs, acceptance_status="mentioned")
        accepted = compute_paper_score(**base_kwargs, acceptance_status="accepted")
        oral = compute_paper_score(**base_kwargs, acceptance_status="oral")
        workshop = compute_paper_score(**base_kwargs, acceptance_status="workshop")

        self.assertEqual(no_venue, mentioned)
        self.assertGreater(workshop, mentioned)
        self.assertGreater(accepted, workshop)
        self.assertGreater(oral, accepted)

    def test_interest_similarity_shifts_score_both_ways(self):
        today = date.today()
        base_kwargs = {
            "match_types": ["Title"],
            "matched_terms_count": 1,
            "publication_dt": today,
            "resource_count": 0,
        }
        neutral = compute_paper_score(**base_kwargs)
        none_similarity = compute_paper_score(**base_kwargs, interest_similarity=None)
        liked = compute_paper_score(**base_kwargs, interest_similarity=0.8)
        disliked = compute_paper_score(**base_kwargs, interest_similarity=-0.8)

        self.assertEqual(neutral, none_similarity)
        self.assertGreater(liked, neutral)
        self.assertLess(disliked, neutral)

    def test_explain_score_includes_interest_bonus(self):
        from app.services.ranking import explain_score

        breakdown = explain_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date.today(),
            resource_count=0,
            interest_similarity=0.5,
        )
        self.assertEqual(breakdown["interest_bonus"], 6.0)
        without = explain_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date.today(),
            resource_count=0,
        )
        self.assertEqual(without["interest_bonus"], 0.0)

    def test_explain_score_includes_venue_bonus(self):
        from app.services.ranking import explain_score

        breakdown = explain_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date.today(),
            resource_count=0,
            acceptance_status="accepted",
        )
        self.assertEqual(breakdown["venue_bonus"], 8.0)
        without = explain_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date.today(),
            resource_count=0,
        )
        self.assertEqual(without["venue_bonus"], 0.0)


if __name__ == "__main__":
    unittest.main()
