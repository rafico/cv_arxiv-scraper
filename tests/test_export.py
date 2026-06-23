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

    def _add_lagged_paper(self, *, arxiv_id: str, title: str) -> None:
        # arXiv announces papers a few days after their publication date, so a
        # freshly scraped paper carries an older publication_dt. The daily export
        # must include it (it shows in the on-screen inbox) — regression guard
        # against the export drifting from the inbox timeframe window.
        today = date.today()
        db.session.add(
            Paper(
                arxiv_id=arxiv_id,
                title=title,
                authors="Author D",
                link=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_link=f"https://arxiv.org/pdf/{arxiv_id}",
                summary_text="Lagged summary",
                match_type="Title",
                matched_terms=["Vision"],
                paper_score=15.0,
                publication_date=(today - timedelta(days=3)).isoformat(),
                publication_dt=today - timedelta(days=3),
                scraped_date=today.isoformat(),
                scraped_at=now_utc(),
                is_hidden=False,
            )
        )
        db.session.commit()

    def test_html_report_includes_announcement_lagged_paper(self):
        self._add_lagged_paper(arxiv_id="2601.0004", title="Announcement Lagged Export Paper")

        html = generate_html_report(self.app, timeframe="daily")

        self.assertIn("Announcement Lagged Export Paper", html)

    def test_bibtex_export_includes_announcement_lagged_paper(self):
        self._add_lagged_paper(arxiv_id="2601.0005", title="Announcement Lagged Bibtex Paper")

        client = self.app.test_client()
        response = client.get("/api/export/bibtex?timeframe=daily")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Announcement Lagged Bibtex Paper", response.get_data(as_text=True))
