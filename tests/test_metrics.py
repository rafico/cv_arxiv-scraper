from __future__ import annotations

from datetime import date, datetime, timedelta

from app.models import Paper, PaperFeedback, RankingConfig, RecommendationMetric, db
from app.services.metrics import (
    compare_metric_outcomes_across_configs,
    compute_mean_time_to_first_open_hours,
)
from app.services.pipeline import ScoredCandidate, WeightedSumRanker
from app.services.ranking import compute_paper_score, resolve_ranking_preferences
from tests.helpers import FlaskDBTestCase


class MetricsTests(FlaskDBTestCase):
    def _create_paper(
        self,
        *,
        arxiv_id: str,
        title: str,
        authors: str,
        match_type: str,
        matched_terms: list[str],
        publication_dt: date,
        scraped_at: datetime,
        citation_count: int | None = None,
    ) -> Paper:
        paper = Paper(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors,
            link=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_link=f"https://arxiv.org/pdf/{arxiv_id}",
            abstract_text="Test abstract",
            summary_text="Test summary",
            topic_tags=["Vision"],
            categories=["cs.CV"],
            resource_links=[],
            match_type=match_type,
            matched_terms=matched_terms,
            paper_score=0.0,
            feedback_score=0,
            is_hidden=False,
            citation_count=citation_count,
            publication_date=publication_dt.isoformat(),
            publication_dt=publication_dt,
            scraped_date=scraped_at.date().isoformat(),
            scraped_at=scraped_at,
        )
        db.session.add(paper)
        db.session.commit()
        return paper

    def test_active_db_ranking_config_overrides_weight_resolution_for_pipeline_and_legacy_scoring(self):
        cfg = RankingConfig(
            name="db-active",
            weights={
                "author_weight": 90,
                "affiliation_weight": 10,
                "title_weight": 2,
                "ai_weight": 5,
                "citation_weight": 0.05,
                "freshness_half_life_days": 14,
            },
            is_active=True,
        )
        db.session.add(cfg)
        db.session.commit()

        preferences = resolve_ranking_preferences(self.app.config["SCRAPER_CONFIG"])
        self.assertEqual(preferences["Author"], 90.0)
        self.assertEqual(preferences["Title"], 2.0)

        publication_dt = date(2026, 4, 1)
        legacy_score = compute_paper_score(
            match_types=["Author"],
            matched_terms_count=1,
            publication_dt=publication_dt,
            resource_count=0,
            config=self.app.config["SCRAPER_CONFIG"],
        )

        ranker = WeightedSumRanker(config=self.app.config["SCRAPER_CONFIG"])
        ranked = ranker.rank(
            [
                ScoredCandidate(
                    entry_data={
                        "arxiv_id": "2604.1001",
                        "link": "https://arxiv.org/abs/2604.1001",
                        "title": "Author paper",
                        "author": "Jane Doe",
                        "authors_list": ["Jane Doe"],
                        "abstract": "Test abstract",
                        "publication_dt": publication_dt,
                        "publication_date": publication_dt.isoformat(),
                        "resource_links": [],
                        "categories": [],
                        "llm_relevance_score": None,
                        "citation_count": None,
                    },
                    match_types=["Author"],
                    matched_terms=["Jane Doe"],
                )
            ]
        )
        self.assertEqual(ranked[0].score, legacy_score)

    def test_compare_metric_outcomes_across_configs_persists_snapshots(self):
        base_time = datetime(2026, 4, 2, 9, 0, 0)
        saved_paper = self._create_paper(
            arxiv_id="2604.2001",
            title="Tracked author paper",
            authors="Jane Doe",
            match_type="Author",
            matched_terms=["Jane Doe"],
            publication_dt=date(2026, 4, 2),
            scraped_at=base_time,
            citation_count=1,
        )
        cited_paper = self._create_paper(
            arxiv_id="2604.2002",
            title="Highly cited baseline",
            authors="Someone Else",
            match_type="Title",
            matched_terms=["Vision"],
            publication_dt=date(2026, 4, 2),
            scraped_at=base_time,
            citation_count=500,
        )

        db.session.add(
            PaperFeedback(
                paper_id=saved_paper.id,
                action="save",
                created_at=base_time + timedelta(hours=2),
            )
        )
        db.session.add(
            PaperFeedback(
                paper_id=cited_paper.id,
                action="skimmed",
                created_at=base_time + timedelta(hours=6),
            )
        )
        db.session.commit()

        author_focused = RankingConfig(
            name="author-focused",
            weights={"author_weight": 80, "title_weight": 2, "citation_weight": 0.01},
            is_active=False,
        )
        citation_focused = RankingConfig(
            name="citation-focused",
            weights={"author_weight": 1, "title_weight": 1, "citation_weight": 10},
            is_active=False,
        )
        db.session.add(author_focused)
        db.session.add(citation_focused)
        db.session.commit()

        results = compare_metric_outcomes_across_configs(
            [author_focused, citation_focused],
            config=self.app.config["SCRAPER_CONFIG"],
            precision_ks=(1,),
            persist=True,
        )

        self.assertEqual(len(results), 2)
        metrics_by_name = {result["config_snapshot"]["name"]: result["metrics"] for result in results}
        self.assertEqual(metrics_by_name["author-focused"]["precision_at_1"], 1.0)
        self.assertEqual(metrics_by_name["citation-focused"]["precision_at_1"], 0.0)
        self.assertEqual(metrics_by_name["author-focused"]["author_follow_hit_rate"], 1.0)
        self.assertEqual(metrics_by_name["citation-focused"]["author_follow_hit_rate"], 1.0)
        self.assertGreater(metrics_by_name["author-focused"]["mean_time_to_first_open_hours"], 0.0)

        stored_metrics = RecommendationMetric.query.order_by(RecommendationMetric.id).all()
        self.assertEqual(len(stored_metrics), 6)
        self.assertEqual(stored_metrics[0].config_snapshot["source"], "db")
        self.assertIn(stored_metrics[0].config_snapshot["name"], {"author-focused", "citation-focused"})

    def test_compute_mean_time_to_first_open_hours_uses_earliest_open_feedback(self):
        base_time = datetime(2026, 4, 1, 8, 0, 0)
        paper = self._create_paper(
            arxiv_id="2604.3001",
            title="Open metric paper",
            authors="Jane Doe",
            match_type="Author",
            matched_terms=["Jane Doe"],
            publication_dt=date(2026, 4, 1),
            scraped_at=base_time,
        )

        db.session.add(PaperFeedback(paper_id=paper.id, action="shared", created_at=base_time + timedelta(hours=5)))
        db.session.add(PaperFeedback(paper_id=paper.id, action="skimmed", created_at=base_time + timedelta(hours=2)))
        db.session.commit()

        self.assertEqual(compute_mean_time_to_first_open_hours(), 2.0)
