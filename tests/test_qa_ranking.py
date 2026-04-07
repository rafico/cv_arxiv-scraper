"""QA tests for ranking, scoring, feedback boosts, and explanations.

Covers: CVARX-56 (Ranking & Scoring)
- Base scoring by match type (Author ~44, Affiliation ~26, Title ~14)
- Term match count bonus (+3 per term)
- Resource availability bonus (+1.5 per link)
- Citation bonus
- Freshness decay and null publication_dt penalty
- Feedback weights (save +10, priority +20, shared +15, skimmed +2, skip -9, ignore -3)
- Combined rank score formula
- Score explanation endpoint
- Custom ranking weights via RankingConfig
"""

from __future__ import annotations

import math
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from app.models import Paper, RankingConfig, db
from app.services.ranking import (
    FEEDBACK_BOOST,
    FEEDBACK_WEIGHTS,
    HALF_LIFE_DAYS,
    LLM_RELEVANCE_WEIGHT,
    MATCH_TYPE_WEIGHTS,
    RESOURCE_SIGNAL_WEIGHT,
    TERM_MATCH_WEIGHT,
    combined_rank_score,
    compute_feedback_delta,
    compute_paper_score,
    explain_score,
    recency_multiplier,
    recompute_all_paper_scores,
)
from tests.helpers import FlaskDBTestCase


class BaseScoreTests(unittest.TestCase):
    """Test base scoring by match type with default weights."""

    @patch("app.services.ranking.utc_today", return_value=date(2026, 4, 7))
    def test_author_match_base_score(self, _):
        score = compute_paper_score(
            match_types=["Author"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=0,
        )
        # Author=44 + 1 term*3 = 47, recency=1.0
        self.assertAlmostEqual(score, 47.0, places=1)

    @patch("app.services.ranking.utc_today", return_value=date(2026, 4, 7))
    def test_affiliation_match_base_score(self, _):
        score = compute_paper_score(
            match_types=["Affiliation"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=0,
        )
        # Affiliation=26 + 1 term*3 = 29, recency=1.0
        self.assertAlmostEqual(score, 29.0, places=1)

    @patch("app.services.ranking.utc_today", return_value=date(2026, 4, 7))
    def test_title_match_base_score(self, _):
        score = compute_paper_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=0,
        )
        # Title=14 + 1 term*3 = 17, recency=1.0
        self.assertAlmostEqual(score, 17.0, places=1)

    @patch("app.services.ranking.utc_today", return_value=date(2026, 4, 7))
    def test_term_match_bonus(self, _):
        score_1 = compute_paper_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=0,
        )
        score_3 = compute_paper_score(
            match_types=["Title"],
            matched_terms_count=3,
            publication_dt=date(2026, 4, 7),
            resource_count=0,
        )
        # Each additional term adds 3 points
        self.assertAlmostEqual(score_3 - score_1, TERM_MATCH_WEIGHT * 2, places=1)

    @patch("app.services.ranking.utc_today", return_value=date(2026, 4, 7))
    def test_resource_availability_bonus(self, _):
        score_0 = compute_paper_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=0,
        )
        score_2 = compute_paper_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=2,
        )
        self.assertAlmostEqual(score_2 - score_0, RESOURCE_SIGNAL_WEIGHT * 2, places=1)

    @patch("app.services.ranking.utc_today", return_value=date(2026, 4, 7))
    def test_resource_bonus_capped_at_4(self, _):
        score_4 = compute_paper_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=4,
        )
        score_10 = compute_paper_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=10,
        )
        self.assertAlmostEqual(score_4, score_10, places=1)

    @patch("app.services.ranking.utc_today", return_value=date(2026, 4, 7))
    def test_citation_bonus(self, _):
        score_0 = compute_paper_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=0,
            citation_count=0,
        )
        score_100 = compute_paper_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=0,
            citation_count=100,
        )
        self.assertGreater(score_100, score_0)


class RecencyTests(unittest.TestCase):
    """Test freshness decay multiplier."""

    def test_today_has_multiplier_1(self):
        today = date.today()
        self.assertAlmostEqual(recency_multiplier(today, today), 1.0, places=3)

    def test_older_papers_decay(self):
        today = date.today()
        old = today - timedelta(days=30)
        mult = recency_multiplier(old, today)
        self.assertLess(mult, 1.0)
        self.assertGreater(mult, 0.0)

    def test_null_publication_date_penalty(self):
        mult = recency_multiplier(None)
        self.assertAlmostEqual(mult, 0.72, places=2)

    def test_half_life_controls_decay_rate(self):
        today = date.today()
        at_half_life = today - timedelta(days=14)
        mult = recency_multiplier(at_half_life, today, half_life_days=14.0)
        expected = math.exp(-1)
        self.assertAlmostEqual(mult, expected, places=3)


class FeedbackDeltaTests(unittest.TestCase):
    """Test feedback weight values."""

    def test_save_delta(self):
        self.assertEqual(compute_feedback_delta("save"), 10)

    def test_priority_delta(self):
        self.assertEqual(compute_feedback_delta("priority"), 20)

    def test_shared_delta(self):
        self.assertEqual(compute_feedback_delta("shared"), 15)

    def test_skimmed_delta(self):
        self.assertEqual(compute_feedback_delta("skimmed"), 2)

    def test_skip_delta(self):
        self.assertEqual(compute_feedback_delta("skip"), -9)

    def test_ignore_delta(self):
        self.assertEqual(compute_feedback_delta("ignore"), -3)

    def test_unknown_action_delta_zero(self):
        self.assertEqual(compute_feedback_delta("nonexistent"), 0)


class CombinedRankScoreTests(unittest.TestCase):
    """Test combined score formula: paper_score + feedback_score * FEEDBACK_BOOST."""

    def test_formula(self):
        result = combined_rank_score(10.0, 4)
        expected = 10.0 + 4 * FEEDBACK_BOOST
        self.assertAlmostEqual(result, expected, places=3)

    def test_zero_feedback(self):
        self.assertAlmostEqual(combined_rank_score(25.0, 0), 25.0, places=3)

    def test_negative_feedback(self):
        result = combined_rank_score(25.0, -5)
        self.assertLess(result, 25.0)


class ExplainScoreTests(unittest.TestCase):
    """Test score explanation breakdown."""

    @patch("app.services.ranking.utc_today", return_value=date(2026, 4, 7))
    def test_explanation_keys(self, _):
        explanation = explain_score(
            match_types=["Author"],
            matched_terms_count=2,
            publication_dt=date(2026, 4, 7),
            resource_count=1,
            feedback_score=10,
        )
        expected_keys = {
            "match_score",
            "term_score",
            "resource_score",
            "ai_bonus",
            "citation_bonus",
            "recency_multiplier",
            "base_score",
            "feedback_bonus",
            "rank_score",
        }
        self.assertEqual(set(explanation.keys()), expected_keys)

    @patch("app.services.ranking.utc_today", return_value=date(2026, 4, 7))
    def test_explanation_rank_score_equals_base_plus_feedback(self, _):
        explanation = explain_score(
            match_types=["Author"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=0,
            feedback_score=10,
        )
        expected = round(explanation["base_score"] + explanation["feedback_bonus"], 3)
        self.assertAlmostEqual(explanation["rank_score"], expected, places=3)

    @patch("app.services.ranking.utc_today", return_value=date(2026, 4, 7))
    def test_explanation_ai_bonus_when_llm_score_provided(self, _):
        explanation = explain_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=0,
            llm_relevance_score=8.0,
        )
        self.assertGreater(explanation["ai_bonus"], 0)

    @patch("app.services.ranking.utc_today", return_value=date(2026, 4, 7))
    def test_explanation_no_ai_bonus_when_llm_none(self, _):
        explanation = explain_score(
            match_types=["Title"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=0,
            llm_relevance_score=None,
        )
        self.assertEqual(explanation["ai_bonus"], 0)


class ExplainEndpointTests(FlaskDBTestCase):
    """Test the /api/papers/<id>/explain endpoint."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        paper = Paper(
            arxiv_id="2607.5555",
            title="Test Paper",
            authors="Jane Doe",
            link="https://arxiv.org/abs/2607.5555",
            pdf_link="https://arxiv.org/pdf/2607.5555",
            abstract_text="test abstract",
            summary_text="summary",
            topic_tags=["Vision"],
            categories=["cs.CV"],
            match_type="Author",
            matched_terms=["Jane Doe"],
            paper_score=47.0,
            feedback_score=10,
            is_hidden=False,
            publication_date=date.today().isoformat(),
            publication_dt=date.today(),
            scraped_date=date.today().isoformat(),
            scraped_at=None,
        )
        db.session.add(paper)
        db.session.commit()
        self.paper_id = paper.id

    def test_explain_returns_breakdown(self):
        response = self.client.get(f"/api/papers/{self.paper_id}/explain")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("match_score", data)
        self.assertIn("rank_score", data)
        self.assertIn("recency_multiplier", data)

    def test_explain_paper_not_found(self):
        response = self.client.get("/api/papers/99999/explain")
        self.assertEqual(response.status_code, 404)


class CustomRankingWeightsTests(FlaskDBTestCase):
    """Test that RankingConfig overrides default weights."""

    def setUp(self):
        super().setUp()

    @patch("app.services.ranking.utc_today", return_value=date(2026, 4, 7))
    def test_custom_author_weight_changes_score(self, _):
        # Default author weight is 44
        default_score = compute_paper_score(
            match_types=["Author"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=0,
        )

        # Custom weight of 100 for author
        custom_score = compute_paper_score(
            match_types=["Author"],
            matched_terms_count=1,
            publication_dt=date(2026, 4, 7),
            resource_count=0,
            ranking_config={"weights": {"author_weight": 100.0}},
        )
        self.assertGreater(custom_score, default_score)

    def test_recompute_all_paper_scores(self):
        paper = Paper(
            arxiv_id="2607.9999",
            title="Recompute Test",
            authors="Alice",
            link="https://arxiv.org/abs/2607.9999",
            pdf_link="https://arxiv.org/pdf/2607.9999",
            abstract_text="test",
            summary_text="test",
            match_type="Title",
            matched_terms=["Vision"],
            paper_score=999.0,
            feedback_score=0,
            is_hidden=False,
            publication_date=date.today().isoformat(),
            publication_dt=date.today(),
            scraped_date=date.today().isoformat(),
        )
        db.session.add(paper)
        db.session.commit()

        updated = recompute_all_paper_scores(self.app)
        self.assertEqual(updated, 1)

        refreshed = db.session.get(Paper, paper.id)
        self.assertNotAlmostEqual(refreshed.paper_score, 999.0, places=1)


if __name__ == "__main__":
    unittest.main()
