"""Tests for BibTeX generation and export endpoints."""

from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from app.models import Paper, PaperFeedback, db
from app.services.bibtex import (
    _escape_latex,
    _format_bibtex_authors,
    paper_to_bibtex,
    papers_to_bibtex,
)
from tests.helpers import FlaskDBTestCase


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        arxiv_id="2603.12345",
        title="Test Paper on Vision",
        authors="Alice Smith, Bob Jones",
        link="https://arxiv.org/abs/2603.12345",
        pdf_link="https://arxiv.org/pdf/2603.12345",
        abstract_text="An abstract about vision.",
        summary_text="A summary.",
        topic_tags=["vision"],
        categories=["cs.CV"],
        resource_links=[],
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=10.0,
        feedback_score=0,
        is_hidden=False,
        publication_date="2026-03-13",
        scraped_date="2026-03-13",
        publication_dt=date(2026, 3, 13),
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    defaults.update(overrides)
    return Paper(**defaults)


class PaperToBibtexTests(unittest.TestCase):
    def test_basic_paper_conversion(self):
        paper = _make_paper()
        bib = paper_to_bibtex(paper)
        self.assertIn("@article{2603_12345,", bib)
        self.assertIn("title = {Test Paper on Vision}", bib)
        self.assertIn("url = {https://arxiv.org/abs/2603.12345}", bib)
        self.assertIn("year = {2026}", bib)

    def test_author_formatting(self):
        result = _format_bibtex_authors("Alice Smith, Bob Jones")
        self.assertEqual(result, "Smith, Alice and Jones, Bob")

    def test_single_author(self):
        result = _format_bibtex_authors("Alice Smith")
        self.assertEqual(result, "Smith, Alice")

    def test_special_characters_escaped(self):
        paper = _make_paper(
            title="Loss & Accuracy: 100% for #1 model_{x}",
            abstract_text="Testing & escaping ~special^ chars",
        )
        bib = paper_to_bibtex(paper)
        self.assertIn(r"Loss \& Accuracy", bib)
        self.assertIn(r"100\%", bib)
        self.assertIn(r"\#1", bib)
        self.assertIn(r"model\_\{x\}", bib)
        self.assertIn(r"Testing \& escaping", bib)

    def test_missing_optional_fields(self):
        paper = _make_paper(abstract_text="", publication_dt=None)
        bib = paper_to_bibtex(paper)
        self.assertIn("@article{", bib)
        self.assertIn("title = {", bib)
        self.assertNotIn("abstract = {", bib)
        self.assertNotIn("year = {", bib)

    def test_arxiv_id_used_as_cite_key(self):
        paper = _make_paper(arxiv_id="2603.99999")
        bib = paper_to_bibtex(paper)
        self.assertTrue(bib.startswith("@article{2603_99999,"))

    def test_multiple_papers_export(self):
        p1 = _make_paper(arxiv_id="2603.00001", link="https://arxiv.org/abs/2603.00001")
        p2 = _make_paper(arxiv_id="2603.00002", link="https://arxiv.org/abs/2603.00002")
        bib = papers_to_bibtex([p1, p2])
        self.assertEqual(bib.count("@article{"), 2)
        self.assertIn("2603_00001", bib)
        self.assertIn("2603_00002", bib)


class BibtexExportEndpointTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def test_export_returns_bib_content_type(self):
        response = self.client.get("/api/export/bibtex")
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/x-bibtex", response.content_type)

    def test_export_returns_download_header(self):
        response = self.client.get("/api/export/bibtex?timeframe=daily")
        self.assertIn("Content-Disposition", response.headers)
        self.assertIn(".bib", response.headers["Content-Disposition"])

    def test_export_saved_view_only_includes_saved(self):
        p1 = _make_paper(title="Saved Paper", arxiv_id="2603.00001", link="https://arxiv.org/abs/2603.00001")
        p2 = _make_paper(title="Unsaved Paper", arxiv_id="2603.00002", link="https://arxiv.org/abs/2603.00002")
        db.session.add_all([p1, p2])
        db.session.commit()

        db.session.add(PaperFeedback(paper_id=p1.id, action="save"))
        db.session.commit()

        response = self.client.get("/api/export/bibtex?view=saved&timeframe=all")
        bib = response.get_data(as_text=True)
        self.assertIn("Saved Paper", bib)
        self.assertNotIn("Unsaved Paper", bib)

    def test_export_empty_returns_empty_bib(self):
        response = self.client.get("/api/export/bibtex?timeframe=all")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "")

    def test_single_paper_bibtex(self):
        paper = _make_paper()
        db.session.add(paper)
        db.session.commit()

        response = self.client.get(f"/api/papers/{paper.id}/bibtex")
        self.assertEqual(response.status_code, 200)
        self.assertIn("@article{", response.get_data(as_text=True))

    def test_single_paper_bibtex_404(self):
        response = self.client.get("/api/papers/99999/bibtex")
        self.assertEqual(response.status_code, 404)


class EscapeLatexTests(unittest.TestCase):
    def test_ampersand(self):
        self.assertEqual(_escape_latex("A & B"), r"A \& B")

    def test_percent(self):
        self.assertEqual(_escape_latex("100%"), r"100\%")

    def test_no_special_chars(self):
        self.assertEqual(_escape_latex("Normal text"), "Normal text")
