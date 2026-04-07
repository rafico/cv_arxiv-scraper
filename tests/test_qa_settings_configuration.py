from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from app.models import DigestRun, Paper, ScrapeRun, db
from tests.helpers import FlaskDBTestCase


def _make_paper(**overrides) -> Paper:
    today = date.today()
    defaults = dict(
        arxiv_id="2604.08001",
        title="Settings QA Paper",
        authors="Jane Doe",
        link="https://arxiv.org/abs/2604.08001",
        pdf_link="https://arxiv.org/pdf/2604.08001",
        abstract_text="A paper used to verify settings recomputation.",
        summary_text="Settings QA summary.",
        topic_tags=["vision"],
        categories=["cs.CV"],
        match_type="Author",
        matched_terms=["Jane Doe"],
        paper_score=47.0,
        feedback_score=0,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    defaults.update(overrides)
    return Paper(**defaults)


class SettingsConfigurationQaTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/settings")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_settings_monitoring_shows_latest_eight_scrape_and_digest_runs(self):
        base = datetime(2026, 4, 7, 12, 0, 0)
        for idx in range(9):
            db.session.add(
                ScrapeRun(
                    status="success" if idx % 2 == 0 else "error",
                    forced=idx == 8,
                    started_at=base + timedelta(minutes=idx),
                    finished_at=base + timedelta(minutes=idx, seconds=30),
                )
            )
            db.session.add(
                DigestRun(
                    status="preview" if idx % 2 == 0 else "success",
                    recipient="user@example.com",
                    subject=f"Digest {idx}",
                    papers_count=idx,
                    preview_only=idx % 2 == 0,
                    started_at=base + timedelta(minutes=idx),
                    finished_at=base + timedelta(minutes=idx, seconds=45),
                )
            )
        db.session.commit()

        response = self.client.get("/settings?section=automation")
        text = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Recent Automation Runs", text)
        self.assertIn("2026-04-07 12:08", text)
        self.assertNotIn("2026-04-07 12:00", text)

    def test_saving_ranking_preferences_recomputes_existing_scores(self):
        paper = _make_paper()
        db.session.add(paper)
        db.session.commit()
        original_score = paper.paper_score

        response = self.client.post(
            "/settings/preferences",
            data={
                "csrf_token": self._csrf_token(),
                "pref_author_weight": "60",
                "pref_affiliation_weight": "26",
                "pref_title_weight": "14",
                "pref_ai_weight": "5",
                "pref_citation_weight": "0.05",
                "pref_freshness_half_life_days": "14",
            },
        )

        self.assertEqual(response.status_code, 302)
        db.session.expire_all()
        refreshed = db.session.get(Paper, paper.id)
        self.assertGreater(refreshed.paper_score, original_score)

        config_path = Path(self.app.config["CONFIG_PATH"])
        saved_config = config_path.read_text(encoding="utf-8")
        self.assertIn("author_weight: 60.0", saved_config)
