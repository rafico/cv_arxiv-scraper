from datetime import date, timedelta
import unittest

from app.services.ranking import (
    combined_rank_score,
    compute_feedback_delta,
    compute_paper_score,
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
        self.assertEqual(compute_feedback_delta("upvote"), 5)
        self.assertEqual(compute_feedback_delta("save"), 7)
        self.assertEqual(compute_feedback_delta("skip"), -9)
        self.assertEqual(compute_feedback_delta("unknown"), 0)

    def test_combined_rank_score(self):
        self.assertEqual(combined_rank_score(10.0, 4), 15.0)


if __name__ == "__main__":
    unittest.main()
