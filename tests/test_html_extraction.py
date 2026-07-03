"""Tests for arXiv-HTML-first section extraction (app/services/html_extraction.py).

All network is mocked. Covers: fixture HTML -> canonical sections + link extraction,
404/garbage -> None, the HTML-first orchestrator's PDF fallback, and the scrape-engine
hook staying non-fatal with the expected (type, text, order_index) output shape.
"""

from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch

from app.models import Paper, PaperSection, db
from app.services import html_extraction
from app.services.html_extraction import (
    HtmlSections,
    SectionExtraction,
    extract_html_sections,
    extract_sections,
)
from app.services.pdf_extraction import ExtractedSection
from tests.helpers import FlaskDBTestCase

FIXTURE_HTML = """
<!DOCTYPE html>
<html><head><style>.x{color:red}</style><script>var y=1;</script></head>
<body>
<h1 class="ltx_title">A Great Paper Title</h1>
<div class="ltx_abstract"><h6 class="ltx_title">Abstract</h6>
<p>We present a novel approach to widget detection using
<math alttext="\\theta_i"><mi>θ</mi></math> parameters.</p></div>
<section><h2>1 Introduction</h2>
<p>Object detection is important. Code available at
<a href="https://github.com/acme/widget">our repository</a>.</p>
<h3>1.1 Scope</h3><p>We focus on widgets specifically.</p>
</section>
<section><h2>2 Method</h2><p>Our method uses a transformer backbone.</p></section>
<section><h2>3 Experiments</h2>
<p>We evaluate on COCO. Project at <a href="https://acme.github.io/widget">site</a>.
See also <a href="https://example.com/news">a news article</a>.</p></section>
<section><h2>4. Conclusion</h2><p>We presented a novel approach.</p></section>
<section><h2>References</h2><p>[1] Some reference.</p></section>
</body></html>
"""


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class TestExtractHtmlSections(unittest.TestCase):
    def _extract(self, html_text: str) -> HtmlSections | None:
        with patch.object(html_extraction, "request_with_backoff", return_value=_FakeResponse(html_text)):
            return extract_html_sections("2601.00001")

    def test_maps_headings_to_canonical_types_in_order(self):
        result = self._extract(FIXTURE_HTML)
        assert result is not None
        types = [section_type for section_type, _text, _idx in result.sections]
        assert types == ["abstract", "introduction", "method", "experiments", "conclusion", "references"]
        # order_index is contiguous and increasing.
        assert [idx for _t, _txt, idx in result.sections] == list(range(len(result.sections)))

    def test_section_text_and_mathml_alttext(self):
        result = self._extract(FIXTURE_HTML)
        assert result is not None
        by_type = {section_type: text for section_type, text, _idx in result.sections}
        # Math is rendered from alttext (source LaTeX), not the noisy MathML tree.
        assert "\\theta_i" in by_type["abstract"]
        assert "novel approach to widget detection" in by_type["abstract"]
        # A non-canonical subsection heading stays as body content of its section.
        assert "Scope" in by_type["introduction"]
        assert "focus on widgets" in by_type["introduction"]
        # script/style content never leaks into sections.
        assert "color:red" not in by_type["abstract"]
        assert "var y" not in by_type["introduction"]

    def test_extracts_only_code_project_links(self):
        result = self._extract(FIXTURE_HTML)
        assert result is not None
        urls = {link["url"] for link in result.links}
        assert "https://github.com/acme/widget" in urls
        assert "https://acme.github.io/widget" in urls
        # A plain news link is not a code/project resource.
        assert "https://example.com/news" not in urls
        types = {link["type"] for link in result.links}
        assert types <= {"code", "dataset", "demo", "project"}

    def test_output_shape_matches_indexer(self):
        # The section indexer consumes (paper_id, section_type, text); rows are built
        # from (section_type, text, order_index) 3-tuples of (str, str, int).
        result = self._extract(FIXTURE_HTML)
        assert result is not None
        for item in result.sections:
            section_type, text, order_index = item
            assert isinstance(section_type, str)
            assert isinstance(text, str)
            assert isinstance(order_index, int)

    def test_404_returns_none(self):
        import requests

        with patch.object(html_extraction, "request_with_backoff", side_effect=requests.HTTPError("404")):
            assert extract_html_sections("0000.00000") is None

    def test_garbage_html_returns_none(self):
        # A page with no recognizable canonical headings is a clean miss.
        assert self._extract("<html><body><p>just some text, no headings</p></body></html>") is None

    def test_parse_failure_returns_none(self):
        class _Boom:
            @property
            def text(self):
                raise ValueError("boom")

        with patch.object(html_extraction, "request_with_backoff", return_value=_Boom()):
            assert extract_html_sections("2601.00002") is None


class TestOrchestrator(unittest.TestCase):
    def test_prefers_html_when_available(self):
        html = HtmlSections(sections=[("abstract", "hi", 0)], links=[{"type": "code", "label": "Code", "url": "u"}])
        with (
            patch.object(html_extraction, "extract_html_sections", return_value=html) as mock_html,
            patch("app.services.pdf_extraction.extract_sections") as mock_pdf,
        ):
            result = extract_sections("2601.1", pdf_content=b"%PDF-x", isolate_pdf=False)
        assert isinstance(result, SectionExtraction)
        assert result.source == "html"
        assert result.sections == [("abstract", "hi", 0)]
        assert result.links and result.links[0]["url"] == "u"
        mock_html.assert_called_once()
        mock_pdf.assert_not_called()

    def test_falls_back_to_pdf_when_html_is_none(self):
        with (
            patch.object(html_extraction, "extract_html_sections", return_value=None),
            patch(
                "app.services.pdf_extraction.extract_sections",
                return_value=[ExtractedSection(section_type="method", text="m", order_index=0)],
            ) as mock_pdf,
        ):
            result = extract_sections("2601.2", pdf_content=b"%PDF-x", isolate_pdf=False)
        assert result.source == "pdf"
        assert result.sections == [("method", "m", 0)]
        assert result.links == []
        mock_pdf.assert_called_once()

    def test_downloads_pdf_when_only_link_given(self):
        with (
            patch.object(html_extraction, "extract_html_sections", return_value=None),
            patch.object(html_extraction, "_download_pdf", return_value=b"%PDF-bytes") as mock_dl,
            patch(
                "app.services.pdf_extraction.extract_sections",
                return_value=[ExtractedSection(section_type="results", text="r", order_index=0)],
            ),
        ):
            result = extract_sections("2601.3", pdf_link="https://x/pdf", isolate_pdf=False)
        assert result.source == "pdf"
        assert result.sections == [("results", "r", 0)]
        mock_dl.assert_called_once()

    def test_total_miss_returns_none_source(self):
        with (
            patch.object(html_extraction, "extract_html_sections", return_value=None),
            patch("app.services.pdf_extraction.extract_sections", return_value=[]),
        ):
            result = extract_sections("2601.4", pdf_content=b"%PDF-x", isolate_pdf=False)
        assert result.source == "none"
        assert result.sections == []


def _make_paper(link: str, arxiv_id: str) -> Paper:
    today = date.today()
    return Paper(
        arxiv_id=arxiv_id,
        title="Paper",
        authors="Author A",
        link=link,
        pdf_link=link.replace("abs", "pdf"),
        abstract_text="abstract",
        summary_text="summary",
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=10.0,
        feedback_score=0,
        is_hidden=False,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )


def _fake_run_isolated(func, *args, **kwargs):
    pdfs = args[0]
    return [[("introduction", f"pdf-{pdf.decode()}", 0)] for pdf in pdfs]


class TestScrapeEngineHook(FlaskDBTestCase):
    """The HTML-first hook in scrape_engine stays non-fatal and persists the shape
    the section index expects."""

    def _run(self, results):
        from app.services.scrape_engine import _extract_sections

        self.app.config["SCRAPER_CONFIG"]["scraper"]["extract_sections"] = True
        _extract_sections(self.app, results)

    @patch("app.services.embeddings.get_embedding_service", side_effect=RuntimeError("skip embeddings"))
    def test_uses_html_when_available(self, _emb):
        link = "https://arxiv.org/abs/2601.5001"
        paper = _make_paper(link, "2601.5001")
        db.session.add(paper)
        db.session.commit()
        paper_id = paper.id

        html = HtmlSections(sections=[("abstract", "from-html", 0), ("method", "m", 1)], links=[])
        results = [{"link": link, "arxiv_id": "2601.5001", "pdf_content": b"PDF1"}]
        with patch("app.services.html_extraction.extract_html_sections", return_value=html):
            self._run(results)

        rows = PaperSection.query.filter_by(paper_id=paper_id).order_by(PaperSection.order_index).all()
        assert [r.section_type for r in rows] == ["abstract", "method"]
        assert rows[0].text == "from-html"

    @patch("app.services.embeddings.get_embedding_service", side_effect=RuntimeError("skip embeddings"))
    @patch("app.services.subprocess_runner.run_isolated", side_effect=_fake_run_isolated)
    def test_falls_back_to_pdf_and_is_non_fatal(self, _run, _emb):
        link = "https://arxiv.org/abs/2601.5002"
        paper = _make_paper(link, "2601.5002")
        db.session.add(paper)
        db.session.commit()
        paper_id = paper.id

        results = [{"link": link, "arxiv_id": "2601.5002", "pdf_content": b"PDF2"}]
        # HTML raising for every paper must not propagate; it degrades to the PDF path.
        with patch("app.services.html_extraction.extract_html_sections", side_effect=RuntimeError("html boom")):
            self._run(results)

        rows = PaperSection.query.filter_by(paper_id=paper_id).all()
        assert len(rows) == 1
        assert rows[0].text == "pdf-PDF2"


if __name__ == "__main__":
    unittest.main()
