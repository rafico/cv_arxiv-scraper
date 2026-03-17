from __future__ import annotations

from datetime import date, timedelta

from app.models import Paper, db
from app.services.export import generate_html_report
from app.services.text import now_utc
from tests.helpers import FlaskDBTestCase


class ExportTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        now = now_utc()
        today = date.today()

        db.session.add(
            Paper(
                arxiv_id="2601.0001",
                title="Visible Recent Paper",
                authors="Author A",
                link="https://arxiv.org/abs/2601.0001",
                pdf_link="https://arxiv.org/pdf/2601.0001",
                summary_text="Recent summary",
                match_type="Title",
                matched_terms=["Vision"],
                paper_score=10.0,
                publication_date=today.isoformat(),
                publication_dt=today,
                scraped_date=today.isoformat(),
                scraped_at=now,
                is_hidden=False,
            )
        )
        db.session.add(
            Paper(
                arxiv_id="2601.0002",
                title="Hidden Paper",
                authors="Author B",
                link="https://arxiv.org/abs/2601.0002",
                pdf_link="https://arxiv.org/pdf/2601.0002",
                summary_text="Hidden summary",
                match_type="Title",
                matched_terms=["Vision"],
                paper_score=20.0,
                publication_date=today.isoformat(),
                publication_dt=today,
                scraped_date=today.isoformat(),
                scraped_at=now,
                is_hidden=True,
            )
        )
        db.session.add(
            Paper(
                arxiv_id="2601.0003",
                title="Older Paper",
                authors="Author C",
                link="https://arxiv.org/abs/2601.0003",
                pdf_link="https://arxiv.org/pdf/2601.0003",
                summary_text="Older summary",
                match_type="Title",
                matched_terms=["Vision"],
                paper_score=5.0,
                publication_date=(today - timedelta(days=40)).isoformat(),
                publication_dt=today - timedelta(days=40),
                scraped_date=today.isoformat(),
                scraped_at=now,
                is_hidden=False,
            )
        )
        db.session.commit()

    def test_generate_html_report_respects_timeframe_and_hidden_filters(self):
        html = generate_html_report(self.app, timeframe="daily")

        self.assertIn("Visible Recent Paper", html)
        self.assertNotIn("Hidden Paper", html)
        self.assertNotIn("Older Paper", html)

    def test_export_endpoint_returns_html(self):
        client = self.app.test_client()
        response = client.get("/api/export?timeframe=daily&download=1")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.content_type)
        self.assertIn("attachment;", response.headers.get("Content-Disposition", ""))
        self.assertIn("Visible Recent Paper", response.get_data(as_text=True))
