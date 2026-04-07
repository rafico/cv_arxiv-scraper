"""QA tests for BibTeX and HTML export.

Covers: CVARX-61 (Export)
- BibTeX: field completeness, special character escaping, single/bulk export
- HTML report: timeframe filtering, download mode, content verification
"""

from __future__ import annotations

import tempfile
from datetime import date, datetime, timedelta, timezone

from app.models import Paper, PaperFeedback, db
from app.services.bibtex import paper_to_bibtex
from app.services.export import generate_html_report
from tests.helpers import FlaskDBTestCase


def _make_paper(idx: int = 0, **overrides) -> Paper:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    defaults = dict(
        arxiv_id=f"2607.{3000 + idx:04d}",
        title=f"Export Test Paper {idx}",
        authors="Alice Smith, Bob Jones",
        link=f"https://arxiv.org/abs/2607.{3000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2607.{3000 + idx:04d}",
        abstract_text="Abstract about vision transformers",
        summary_text="Summary text",
        topic_tags=["Vision"],
        categories=["cs.CV"],
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=10.0 + idx,
        feedback_score=0,
        is_hidden=False,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=now,
    )
    defaults.update(overrides)
    return Paper(**defaults)


class BibtexFieldCompletenessTests(FlaskDBTestCase):
    """Verify all expected BibTeX fields are populated."""

    def test_all_fields_present(self):
        paper = _make_paper(0)
        db.session.add(paper)
        db.session.commit()

        bib = paper_to_bibtex(paper)
        self.assertIn("@article{", bib)
        self.assertIn("author = {", bib)
        self.assertIn("title = {", bib)
        self.assertIn("year = {", bib)
        self.assertIn("url = {", bib)
        self.assertIn("eprint = {", bib)
        self.assertIn("archiveprefix = {arXiv}", bib)
        self.assertIn("pdf = {", bib)
        self.assertIn("abstract = {", bib)

    def test_bibtex_author_format_last_first(self):
        paper = _make_paper(0, authors="Alice Smith, Bob Jones")
        bib = paper_to_bibtex(paper)
        self.assertIn("Smith, Alice and Jones, Bob", bib)

    def test_bibtex_special_char_escaping(self):
        paper = _make_paper(0, title="10% Improvement & Results for #1 Model")
        bib = paper_to_bibtex(paper)
        self.assertIn(r"10\% Improvement \& Results for \#1 Model", bib)

    def test_bibtex_missing_abstract_excluded(self):
        paper = _make_paper(0, abstract_text="")
        bib = paper_to_bibtex(paper)
        self.assertNotIn("abstract = {", bib)

    def test_bibtex_missing_date_excludes_year(self):
        paper = _make_paper(0, publication_dt=None)
        bib = paper_to_bibtex(paper)
        self.assertNotIn("year = {", bib)


class BibtexEndpointTests(FlaskDBTestCase):
    """Test BibTeX export API endpoints."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_single_paper_bibtex_endpoint(self):
        p = _make_paper(0)
        db.session.add(p)
        db.session.commit()

        response = self.client.get(f"/api/papers/{p.id}/bibtex")
        self.assertEqual(response.status_code, 200)
        self.assertIn("@article{", response.get_data(as_text=True))

    def test_bulk_bibtex_endpoint(self):
        p1 = _make_paper(0)
        p2 = _make_paper(1)
        db.session.add_all([p1, p2])
        db.session.commit()

        response = self.client.get(f"/api/papers/bulk-bibtex?ids={p1.id},{p2.id}")
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        self.assertEqual(text.count("@article{"), 2)

    def test_bibtex_export_all_timeframe(self):
        p = _make_paper(0)
        db.session.add(p)
        db.session.commit()

        response = self.client.get("/api/export/bibtex?timeframe=all")
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/x-bibtex", response.content_type)
        self.assertIn("@article{", response.get_data(as_text=True))

    def test_bibtex_export_saved_view(self):
        p1 = _make_paper(0, title="Saved Export")
        p2 = _make_paper(1, title="Not Saved Export")
        db.session.add_all([p1, p2])
        db.session.commit()
        db.session.add(PaperFeedback(paper_id=p1.id, action="save"))
        db.session.commit()

        response = self.client.get("/api/export/bibtex?view=saved&timeframe=all")
        text = response.get_data(as_text=True)
        self.assertIn("Saved Export", text)
        self.assertNotIn("Not Saved Export", text)


class HtmlReportTests(FlaskDBTestCase):
    """Test HTML report generation."""

    def setUp(self):
        super().setUp()
        today = date.today()

        db.session.add(_make_paper(0, title="Recent Paper"))
        db.session.add(
            _make_paper(
                1,
                title="Old Paper",
                publication_date=(today - timedelta(days=40)).isoformat(),
                publication_dt=today - timedelta(days=40),
            )
        )
        db.session.add(_make_paper(2, title="Hidden Paper", is_hidden=True))
        db.session.commit()

    def test_daily_report_excludes_old_papers(self):
        html = generate_html_report(self.app, timeframe="daily")
        self.assertIn("Recent Paper", html)
        self.assertNotIn("Old Paper", html)

    def test_daily_report_excludes_hidden_papers(self):
        html = generate_html_report(self.app, timeframe="daily")
        self.assertNotIn("Hidden Paper", html)

    def test_all_timeframe_includes_old_papers(self):
        html = generate_html_report(self.app, timeframe="all")
        self.assertIn("Recent Paper", html)
        self.assertIn("Old Paper", html)

    def test_weekly_timeframe(self):
        html = generate_html_report(self.app, timeframe="weekly")
        self.assertIn("Recent Paper", html)
        self.assertNotIn("Old Paper", html)

    def test_monthly_timeframe_includes_old_within_30_days(self):
        html = generate_html_report(self.app, timeframe="monthly")
        self.assertIn("Recent Paper", html)

    def test_output_path_writes_file(self):
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        generate_html_report(self.app, timeframe="daily", output_path=path)
        content = Path(path).read_text()
        self.assertIn("Recent Paper", content)
        Path(path).unlink()

    def test_invalid_timeframe_defaults_to_daily(self):
        html = generate_html_report(self.app, timeframe="invalid")
        # Should not crash, defaults to daily
        self.assertIn("Recent Paper", html)

    def test_export_endpoint_download_mode(self):
        client = self.app.test_client()
        response = client.get("/api/export?timeframe=daily&download=1")
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment;", response.headers.get("Content-Disposition", ""))

    def test_export_endpoint_inline_mode(self):
        client = self.app.test_client()
        response = client.get("/api/export?timeframe=daily")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.content_type)


if __name__ == "__main__":
    import unittest

    unittest.main()
