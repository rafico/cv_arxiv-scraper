import unittest
from datetime import date, datetime, timezone
from types import SimpleNamespace

from app.models import Paper, db
from app.services.implementation_readiness import (
    PARTIAL_THRESHOLD,
    RUNNABLE_THRESHOLD,
    ReadinessResult,
    implementation_readiness,
    readiness_badge,
    readiness_bonus,
)
from app.services.ranking import compute_paper_score, explain_score
from tests.helpers import FlaskDBTestCase

TODAY = date(2026, 7, 3)


def _paper(**overrides):
    """Lightweight paper-like object for the pure scoring function."""
    base = dict(
        github_repo="octocat/demo",
        github_stars=None,
        github_license=None,
        resource_links=[],
        hf_upvotes=None,
        scraped_at=datetime(2026, 7, 3),
        publication_dt=None,
        archived=None,
        pushed_at=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class ReadinessScoringTests(unittest.TestCase):
    def test_more_stars_scores_higher(self):
        low = implementation_readiness(_paper(github_stars=5), today=TODAY)
        high = implementation_readiness(_paper(github_stars=5000), today=TODAY)
        self.assertGreater(high.score, low.score)

    def test_license_ordering_permissive_beats_known_beats_none(self):
        none_lic = implementation_readiness(_paper(github_license=None), today=TODAY)
        known = implementation_readiness(_paper(github_license="GPL-3.0"), today=TODAY)
        permissive = implementation_readiness(_paper(github_license="MIT"), today=TODAY)
        self.assertGreater(permissive.score, known.score)
        self.assertGreater(known.score, none_lic.score)

    def test_noassertion_license_counts_as_none(self):
        none_lic = implementation_readiness(_paper(github_license=None), today=TODAY)
        noassertion = implementation_readiness(_paper(github_license="NOASSERTION"), today=TODAY)
        self.assertEqual(noassertion.score, none_lic.score)

    def test_fresher_push_scores_higher(self):
        fresh = implementation_readiness(_paper(pushed_at="2026-07-01T00:00:00Z"), today=TODAY)
        stale = implementation_readiness(_paper(pushed_at="2023-01-01T00:00:00Z"), today=TODAY)
        self.assertGreater(fresh.score, stale.score)

    def test_archived_repo_scores_lower(self):
        active = implementation_readiness(_paper(archived=False), today=TODAY)
        archived = implementation_readiness(_paper(archived=True), today=TODAY)
        self.assertGreater(active.score, archived.score)
        self.assertIn("Repository archived", archived.reasons)

    def test_recent_push_reason_only_from_pushed_at(self):
        # Freshness from the scraped_at fallback must not claim "Recently updated".
        from_scrape = implementation_readiness(_paper(scraped_at=datetime(2026, 7, 3)), today=TODAY)
        self.assertNotIn("Recently updated", from_scrape.reasons)
        from_push = implementation_readiness(_paper(pushed_at="2026-06-25T00:00:00Z"), today=TODAY)
        self.assertIn("Recently updated", from_push.reasons)

    def test_score_bounded_0_100(self):
        loaded = implementation_readiness(
            _paper(
                github_stars=99999,
                github_license="Apache-2.0",
                pushed_at="2026-07-03T00:00:00Z",
                resource_links=[{"type": "project", "url": "https://example.com"}],
                hf_upvotes=999,
            ),
            today=TODAY,
        )
        self.assertLessEqual(loaded.score, 100)
        self.assertGreaterEqual(loaded.score, 0)


class ReadinessTierTests(unittest.TestCase):
    def test_loaded_repo_is_runnable(self):
        result = implementation_readiness(
            _paper(github_stars=1200, github_license="MIT", pushed_at="2026-06-20T00:00:00Z"),
            today=TODAY,
        )
        self.assertEqual(result.tier, "runnable")
        self.assertGreaterEqual(result.score, RUNNABLE_THRESHOLD)

    def test_bare_fresh_repo_is_partial(self):
        result = implementation_readiness(_paper(scraped_at=datetime(2026, 7, 3)), today=TODAY)
        self.assertEqual(result.tier, "partial")
        self.assertGreaterEqual(result.score, PARTIAL_THRESHOLD)
        self.assertLess(result.score, RUNNABLE_THRESHOLD)

    def test_no_repo_is_none_even_with_secondary_signals(self):
        result = implementation_readiness(
            _paper(
                github_repo=None,
                resource_links=[{"type": "project", "url": "https://example.com"}],
                hf_upvotes=100,
            ),
            today=TODAY,
        )
        self.assertEqual(result.tier, "none")

    def test_result_is_readiness_result(self):
        result = implementation_readiness(_paper(), today=TODAY)
        self.assertIsInstance(result, ReadinessResult)
        self.assertIsInstance(result.reasons, list)


class ReadinessBadgeTests(unittest.TestCase):
    def test_badge_present_for_runnable(self):
        badge = readiness_badge(
            _paper(github_stars=1200, github_license="MIT", pushed_at="2026-06-20T00:00:00Z"),
            today=TODAY,
        )
        self.assertIsNotNone(badge)
        assert badge is not None  # narrow for type-checkers
        self.assertEqual(badge["label"], "Runnable")
        self.assertIn("title", badge)

    def test_badge_absent_for_partial(self):
        self.assertIsNone(readiness_badge(_paper(scraped_at=datetime(2026, 7, 3)), today=TODAY))

    def test_badge_absent_for_no_repo(self):
        self.assertIsNone(readiness_badge(_paper(github_repo=None), today=TODAY))


class ReadinessRankingBonusTests(unittest.TestCase):
    def test_readiness_bonus_helper_bounded(self):
        self.assertEqual(readiness_bonus(None, 2.0), 0.0)
        self.assertEqual(readiness_bonus(0, 2.0), 0.0)
        self.assertEqual(readiness_bonus(100, 2.0), 2.0)
        self.assertAlmostEqual(readiness_bonus(50, 2.0), 1.0)
        # Never exceeds the weight even for an out-of-range score.
        self.assertLessEqual(readiness_bonus(150, 2.0), 2.0)

    def test_higher_readiness_raises_paper_score(self):
        kwargs = dict(match_types=["Title"], matched_terms_count=1, publication_dt=date.today(), resource_count=0)
        without = compute_paper_score(**kwargs, readiness_score=0)
        with_readiness = compute_paper_score(**kwargs, readiness_score=100)
        self.assertGreater(with_readiness, without)
        # Bounded by the (small) readiness weight at recency 1.0.
        self.assertLessEqual(with_readiness - without, 2.0 + 1e-6)

    def test_explain_score_surfaces_bounded_readiness_bonus(self):
        kwargs = dict(match_types=["Title"], matched_terms_count=1, publication_dt=date.today(), resource_count=0)
        breakdown = explain_score(**kwargs, readiness_score=80)
        self.assertGreater(breakdown["readiness_bonus"], 0)
        self.assertLessEqual(breakdown["readiness_bonus"], 2.0)

    def test_explain_score_zero_readiness_bonus_when_absent(self):
        kwargs = dict(match_types=["Title"], matched_terms_count=1, publication_dt=date.today(), resource_count=0)
        breakdown = explain_score(**kwargs, readiness_score=None)
        self.assertEqual(breakdown["readiness_bonus"], 0.0)


class RunnableFilterDashboardTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        today = date.today()

        def add(title, **fields):
            paper = Paper(
                arxiv_id=fields.pop("arxiv_id"),
                title=title,
                authors="Author A",
                link=f"https://arxiv.org/abs/{title}",
                pdf_link=f"https://arxiv.org/pdf/{title}",
                abstract_text="abstract",
                summary_text="summary",
                match_type="Title",
                matched_terms=["vision"],
                paper_score=10.0,
                publication_date=today.isoformat(),
                publication_dt=today,
                scraped_date=today.isoformat(),
                scraped_at=now,
                **fields,
            )
            db.session.add(paper)

        add(
            "RunnablePaper",
            arxiv_id="2607.0001",
            github_repo="octocat/runnable",
            github_stars=500,
            github_license="MIT",
        )
        add("PartialPaper", arxiv_id="2607.0002", github_repo="octocat/bare")  # repo only -> partial
        add("NoCodePaper", arxiv_id="2607.0003")  # no repo -> none
        db.session.commit()
        self.client = self.app.test_client()

    def test_runnable_filter_returns_only_runnable_papers(self):
        response = self.client.get("/?resource_filter=runnable&timeframe=all&view=inbox")
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        self.assertIn("RunnablePaper", text)
        self.assertNotIn("PartialPaper", text)
        self.assertNotIn("NoCodePaper", text)

    def test_runnable_badge_rendered_for_runnable_paper(self):
        # Unfiltered view so the assertion targets the card badge, not the filter
        # dropdown option (which always contains the word "Runnable").
        response = self.client.get("/?timeframe=all&view=inbox")
        text = response.get_data(as_text=True)
        self.assertIn("⚙ Runnable", text)

    def test_unfiltered_view_still_lists_all_papers(self):
        response = self.client.get("/?timeframe=all&view=inbox")
        text = response.get_data(as_text=True)
        self.assertIn("RunnablePaper", text)
        self.assertIn("PartialPaper", text)
        self.assertIn("NoCodePaper", text)


if __name__ == "__main__":
    unittest.main()
