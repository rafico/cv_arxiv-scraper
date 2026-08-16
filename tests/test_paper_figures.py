"""Tests for inline figure previews: extraction, pipeline hook, serving, UI, backfill."""

from __future__ import annotations

import io
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from app.models import Paper, db
from app.services.thumbnail_generator import (
    MAX_PAPER_FIGURES,
    _decode_and_save_figures,
    extract_pdf_figures,
    figure_paths_for,
    generate_paper_figures,
    parse_figure_image_urls,
)
from tests.helpers import FlaskDBTestCase

FIXTURE_HTML = """
<html><body>
<img src="outside-any-figure.png">
<figure class="ltx_figure">
  <img class="ltx_graphics" src="x1.png">
  <figcaption>Figure 1: teaser.</figcaption>
</figure>
<figure class="ltx_figure">
  <img src="extracted/fig2.jpg">
  <img src="diagram.svg">
  <img src="data:image/png;base64,AAAA">
</figure>
<figure><img src="https://other.example/abs.webp"></figure>
</body></html>
"""

BASE_URL = "https://arxiv.org/html/2501.00001v1"


def _png_bytes(width: int, height: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(30, 90, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def _image_pdf(width_px: int = 600, height_px: int = 400) -> bytes:
    """A real one-page PDF whose page is a single embedded image (via Pillow)."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width_px, height_px), color=(120, 40, 200)).save(buffer, format="PDF")
    return buffer.getvalue()


# ── HTML parsing ──────────────────────────────────────────────────────────────


def test_parse_figure_image_urls_resolves_and_filters():
    urls = parse_figure_image_urls(FIXTURE_HTML, BASE_URL)

    assert urls == [
        f"{BASE_URL}/x1.png",
        f"{BASE_URL}/extracted/fig2.jpg",
        "https://other.example/abs.webp",
    ]


def test_parse_figure_image_urls_ignores_images_outside_figures():
    urls = parse_figure_image_urls('<img src="banner.png"><p>no figures</p>', BASE_URL)
    assert urls == []


def test_parse_figure_image_urls_handles_nested_figures_and_dedup():
    html = '<figure><figure><img src="a.png"></figure><img src="a.png"><img src="b.png"></figure>'
    urls = parse_figure_image_urls(html, f"{BASE_URL}/")
    assert urls == [f"{BASE_URL}/a.png", f"{BASE_URL}/b.png"]


# ── Decoding / saving ─────────────────────────────────────────────────────────


def test_decode_and_save_skips_small_and_broken_images(tmp_path):
    out_paths = [str(tmp_path / f"p_fig{i}.png") for i in (1, 2)]
    blobs = [
        _png_bytes(20, 20),  # icon-sized: skipped
        b"not an image",  # undecodable: skipped
        _png_bytes(400, 300),  # kept
    ]

    saved = _decode_and_save_figures(blobs, out_paths)

    assert saved == 1
    assert Path(out_paths[0]).exists()
    assert not Path(out_paths[1]).exists()
    # No temp-file droppings.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p_fig1.png"]


def test_decode_and_save_respects_out_path_cap(tmp_path):
    out_paths = [str(tmp_path / "p_fig1.png")]
    saved = _decode_and_save_figures([_png_bytes(400, 300), _png_bytes(500, 300)], out_paths)
    assert saved == 1


# ── PDF fallback ──────────────────────────────────────────────────────────────


def test_extract_pdf_figures_crops_embedded_image(tmp_path):
    out_paths = [str(tmp_path / f"p_fig{i}.png") for i in range(1, MAX_PAPER_FIGURES + 1)]

    saved = extract_pdf_figures(_image_pdf(), out_paths)

    assert saved == 1
    assert Path(out_paths[0]).exists()
    assert Path(out_paths[0]).stat().st_size > 0


def test_extract_pdf_figures_ignores_text_only_pdf(tmp_path):
    import os
    import tempfile

    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 6, txt="A text-only page with no figures.", ln=1)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf.output(tmp.name)
    try:
        content = Path(tmp.name).read_bytes()
    finally:
        os.unlink(tmp.name)

    saved = extract_pdf_figures(content, [str(tmp_path / "p_fig1.png")])

    assert saved == 0
    assert list(tmp_path.iterdir()) == []


# ── generate_paper_figures orchestration ─────────────────────────────────────


def _html_then_images(image_bytes_by_index: dict[int, bytes] | None = None):
    """request_with_backoff side effect: HTML page first, then figure images."""
    calls = {"n": 0}

    def fake_request(method, url, **kwargs):
        calls["n"] += 1
        if url == "https://arxiv.org/html/2501.00001":  # the page itself, not its assets
            return Mock(text=FIXTURE_HTML, url=BASE_URL)
        index = calls["n"] - 2  # 0-based image index
        content = (image_bytes_by_index or {}).get(index, _png_bytes(400, 300))
        return Mock(content=content)

    return fake_request


def test_generate_paper_figures_from_html(tmp_path):
    static_dir = tmp_path / "static"

    with patch("app.services.thumbnail_generator.request_with_backoff", side_effect=_html_then_images()):
        saved = generate_paper_figures("2501.00001", static_dir)

    assert saved == 3  # three raster figure URLs in the fixture
    thumbnails_dir = static_dir / "thumbnails"
    for index in (1, 2, 3):
        assert (thumbnails_dir / f"2501.00001_fig{index}.png").exists()
    assert not (thumbnails_dir / "2501.00001_fig4.png").exists()


def test_generate_paper_figures_skips_small_html_images(tmp_path):
    static_dir = tmp_path / "static"
    side_effect = _html_then_images({1: _png_bytes(30, 30)})  # second image is an icon

    with patch("app.services.thumbnail_generator.request_with_backoff", side_effect=side_effect):
        saved = generate_paper_figures("2501.00001", static_dir)

    assert saved == 2
    thumbnails_dir = static_dir / "thumbnails"
    assert (thumbnails_dir / "2501.00001_fig1.png").exists()
    assert (thumbnails_dir / "2501.00001_fig2.png").exists()
    assert not (thumbnails_dir / "2501.00001_fig3.png").exists()


def test_generate_paper_figures_falls_back_to_pdf_on_missing_html(tmp_path):
    static_dir = tmp_path / "static"

    def no_html(method, url, **kwargs):
        raise requests.HTTPError("404 Client Error: Not Found")

    with patch("app.services.thumbnail_generator.request_with_backoff", side_effect=no_html):
        saved = generate_paper_figures("2401.99999", static_dir, pdf_content=_image_pdf())

    assert saved == 1
    assert (static_dir / "thumbnails" / "2401.99999_fig1.png").exists()


def test_generate_paper_figures_downloads_pdf_when_no_bytes_given(tmp_path):
    static_dir = tmp_path / "static"
    pdf_bytes = _image_pdf()

    def fake_request(method, url, **kwargs):
        if url.startswith("https://arxiv.org/html/"):
            raise requests.HTTPError("404 Client Error: Not Found")
        return Mock(content=pdf_bytes, headers={"Content-Type": "application/pdf"})

    with patch("app.services.thumbnail_generator.request_with_backoff", side_effect=fake_request):
        saved = generate_paper_figures("2401.88888", static_dir, pdf_link="https://arxiv.org/pdf/2401.88888")

    assert saved == 1
    assert (static_dir / "thumbnails" / "2401.88888_fig1.png").exists()


def test_generate_paper_figures_never_raises(tmp_path):
    with patch(
        "app.services.thumbnail_generator.request_with_backoff",
        side_effect=RuntimeError("network down"),
    ):
        saved = generate_paper_figures("2501.00002", tmp_path / "static", pdf_content=b"not a pdf")
    assert saved == 0


def test_generate_paper_figures_is_idempotent(tmp_path):
    static_dir = tmp_path / "static"
    thumbnails_dir = static_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True)
    (thumbnails_dir / "2501.00003_fig1.png").write_bytes(b"png")
    (thumbnails_dir / "2501.00003_fig2.png").write_bytes(b"png")

    with patch("app.services.thumbnail_generator.request_with_backoff") as mock_req:
        saved = generate_paper_figures("2501.00003", static_dir)

    assert saved == 2
    mock_req.assert_not_called()


def test_generate_paper_figures_rejects_traversal_id(tmp_path):
    static_dir = tmp_path / "static"

    with patch("app.services.thumbnail_generator.request_with_backoff") as mock_req:
        saved = generate_paper_figures("../../evil", static_dir)

    assert saved == 0
    mock_req.assert_not_called()
    # Nothing written anywhere under the temp tree.
    assert not (static_dir / "thumbnails").exists()
    assert list(tmp_path.iterdir()) == []


def _http_404() -> requests.HTTPError:
    return requests.HTTPError("404 Client Error: Not Found", response=Mock(status_code=404))


def test_generate_paper_figures_memoizes_conclusive_negative(tmp_path):
    static_dir = tmp_path / "static"

    # No HTML rendition (a real 404) and no PDF source: a stable no-figures fact.
    with patch("app.services.thumbnail_generator.request_with_backoff", side_effect=_http_404()) as mock_req:
        assert generate_paper_figures("2401.77777", static_dir) == 0
        assert mock_req.called
    assert (static_dir / "thumbnails" / "2401.77777_nofig").exists()

    # Second run: sentinel short-circuits before any network call.
    with patch("app.services.thumbnail_generator.request_with_backoff") as mock_req:
        assert generate_paper_figures("2401.77777", static_dir) == 0
        mock_req.assert_not_called()


def test_generate_paper_figures_does_not_memoize_transient_failure(tmp_path):
    static_dir = tmp_path / "static"

    with patch(
        "app.services.thumbnail_generator.request_with_backoff",
        side_effect=RuntimeError("network down"),
    ):
        assert generate_paper_figures("2401.66666", static_dir, pdf_link="https://arxiv.org/pdf/2401.66666") == 0

    assert not (static_dir / "thumbnails" / "2401.66666_nofig").exists()


def test_generate_paper_figures_memoizes_figureless_pdf(tmp_path):
    import os
    import tempfile

    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 6, txt="A text-only page with no figures.", ln=1)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf.output(tmp.name)
    try:
        text_pdf = Path(tmp.name).read_bytes()
    finally:
        os.unlink(tmp.name)

    static_dir = tmp_path / "static"
    with patch("app.services.thumbnail_generator.request_with_backoff", side_effect=_http_404()):
        assert generate_paper_figures("2401.55555", static_dir, pdf_content=text_pdf) == 0

    assert (static_dir / "thumbnails" / "2401.55555_nofig").exists()


# ── figure_paths_for ─────────────────────────────────────────────────────────


def test_figure_paths_for_returns_existing_in_order(tmp_path):
    static_dir = tmp_path / "static"
    thumbnails_dir = static_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True)
    (thumbnails_dir / "2501.00004_fig3.png").write_bytes(b"png")
    (thumbnails_dir / "2501.00004_fig1.png").write_bytes(b"png")

    paths = figure_paths_for("2501.00004", static_dir)

    assert [path.name for path in paths] == ["2501.00004_fig1.png", "2501.00004_fig3.png"]


def test_figure_paths_for_rejects_traversal_id(tmp_path):
    static_dir = tmp_path / "static"
    (static_dir / "thumbnails").mkdir(parents=True)
    # Even if a matching file exists outside the cache dir, a hostile id gets [].
    (tmp_path / "secret_fig1.png").write_bytes(b"png")

    assert figure_paths_for("../secret", static_dir) == []


# ── Scrape-engine hook ────────────────────────────────────────────────────────


class GenerateFiguresHookTests(FlaskDBTestCase):
    def _results(self, count: int) -> list[dict]:
        return [
            {
                "arxiv_id": f"2607.{i:05d}",
                "link": f"https://arxiv.org/abs/2607.{i:05d}",
                "pdf_link": f"https://arxiv.org/pdf/2607.{i:05d}",
                "pdf_content": b"%PDF-fake",
            }
            for i in range(count)
        ]

    def test_hook_is_non_fatal_and_preserves_pdf_content(self):
        from app.services import scrape_engine

        results = self._results(2)
        with patch(
            "app.services.thumbnail_generator.generate_paper_figures",
            side_effect=RuntimeError("boom"),
        ):
            scrape_engine._generate_figures(self.app, results, Mock())

        # pdf_content must survive for the section-extraction step downstream.
        self.assertEqual(results[0]["pdf_content"], b"%PDF-fake")

    def test_hook_caps_papers_per_run(self):
        from app.services import scrape_engine

        results = self._results(scrape_engine._FIGURES_PER_RUN_CAP + 10)
        with patch("app.services.thumbnail_generator.generate_paper_figures", return_value=0) as mock_generate:
            scrape_engine._generate_figures(self.app, results, Mock())

        self.assertEqual(mock_generate.call_count, scrape_engine._FIGURES_PER_RUN_CAP)

    def test_hook_passes_cached_pdf_bytes(self):
        from app.services import scrape_engine

        results = self._results(1)
        with patch("app.services.thumbnail_generator.generate_paper_figures", return_value=1) as mock_generate:
            scrape_engine._generate_figures(self.app, results, Mock())

        kwargs = mock_generate.call_args.kwargs
        self.assertEqual(kwargs["pdf_content"], b"%PDF-fake")
        self.assertEqual(kwargs["pdf_link"], "https://arxiv.org/pdf/2607.00000")


# ── Serving route + dashboard integration ────────────────────────────────────


class FigureRouteTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        self.paper = Paper(
            arxiv_id="2601.00001",
            title="Figure Paper",
            authors="Author A",
            link="https://arxiv.org/abs/2601.00001",
            pdf_link="https://arxiv.org/pdf/2601.00001",
            abstract_text="vision",
            summary_text="Summary",
            match_type="Title",
            matched_terms=["vision"],
            paper_score=10.0,
            publication_date=date.today().isoformat(),
            publication_dt=date.today(),
            scraped_date=date.today().isoformat(),
            scraped_at=now,
        )
        db.session.add(self.paper)
        db.session.commit()

        static_root = Path(self._tmpdir.name) / "static"
        (static_root / "thumbnails").mkdir(parents=True)
        self.app.static_folder = str(static_root)
        self.thumbnails_dir = static_root / "thumbnails"
        self.client = self.app.test_client()

    def _write_figure(self, index: int) -> None:
        (self.thumbnails_dir / f"2601.00001_fig{index}.png").write_bytes(b"\x89PNG\r\n\x1a\nfig")

    def test_serves_existing_figure(self):
        self._write_figure(1)
        response = self.client.get(f"/papers/{self.paper.id}/figures/1.png")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"fig", response.data)

    def test_missing_figure_is_404(self):
        response = self.client.get(f"/papers/{self.paper.id}/figures/1.png")
        self.assertEqual(response.status_code, 404)

    def test_out_of_range_index_is_404(self):
        self._write_figure(1)
        for index in (0, MAX_PAPER_FIGURES + 1):
            response = self.client.get(f"/papers/{self.paper.id}/figures/{index}.png")
            self.assertEqual(response.status_code, 404)

    def test_unknown_paper_is_404(self):
        response = self.client.get("/papers/999999/figures/1.png")
        self.assertEqual(response.status_code, 404)

    def test_dashboard_details_include_figure_strip_when_cached(self):
        self._write_figure(1)
        self._write_figure(2)

        text = self.client.get("/?timeframe=all").get_data(as_text=True)

        self.assertIn(f"/papers/{self.paper.id}/figures/1.png", text)
        self.assertIn(f"/papers/{self.paper.id}/figures/2.png", text)

    def test_dashboard_omits_strip_without_figures(self):
        text = self.client.get("/?timeframe=all").get_data(as_text=True)
        self.assertNotIn("/figures/", text)
        self.assertNotIn("paper-figures", text)


class FigurePartialRenderTests(FlaskDBTestCase):
    def _render(self, paper) -> str:
        from flask import render_template

        with self.app.test_request_context("/"):
            return render_template("partials/_paper_figures.html", paper=paper)

    def test_renders_lazy_images_for_each_index(self):
        html = self._render(SimpleNamespace(id=7, figure_indices=[1, 3]))
        self.assertIn("/papers/7/figures/1.png", html)
        self.assertIn("/papers/7/figures/3.png", html)
        self.assertIn('loading="lazy"', html)

    def test_renders_nothing_without_figures(self):
        self.assertEqual(self._render(SimpleNamespace(id=7, figure_indices=[])).strip(), "")

    def test_renders_nothing_when_attribute_missing(self):
        # Papers rendered outside the dashboard route lack figure_indices entirely.
        self.assertEqual(self._render(SimpleNamespace(id=7)).strip(), "")


# ── Backfill CLI ─────────────────────────────────────────────────────────────


class BackfillFiguresTests(FlaskDBTestCase):
    def _paper(self, arxiv_id: str) -> Paper:
        return Paper(
            arxiv_id=arxiv_id,
            title=f"Paper {arxiv_id}",
            authors="Author A",
            link=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_link=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            match_type="Title",
            matched_terms=["vision"],
            paper_score=1.0,
            publication_date="2026-01-01",
            scraped_date="2026-01-01",
        )

    @patch("app.services.thumbnail_generator.generate_paper_figures", return_value=2)
    def test_backfill_figures_only_targets_papers_without_figures(self, mock_generate):
        from app.cli.backfill import backfill_thumbnails

        static_dir = Path(self._tmpdir.name) / "static"
        self.app.static_folder = str(static_dir)
        db.session.add_all([self._paper("2601.00003"), self._paper("2601.00004")])
        db.session.commit()

        thumbnails_dir = static_dir / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        (thumbnails_dir / "2601.00003_fig1.png").write_bytes(b"png")

        generated = backfill_thumbnails(self.app, batch_size=10, delay_seconds=0, figures=True, emit=lambda _: None)

        self.assertEqual(generated, 1)
        mock_generate.assert_called_once()
        self.assertEqual(mock_generate.call_args.args[0], "2601.00004")
        self.assertEqual(mock_generate.call_args.kwargs["pdf_link"], "https://arxiv.org/pdf/2601.00004.pdf")

    def test_thumbnails_parser_accepts_figures_flag(self):
        from app.cli.backfill import build_parser

        args = build_parser().parse_args(["thumbnails", "--figures", "--batch-size", "5", "--delay", "0"])
        self.assertTrue(args.figures)
        self.assertFalse(args.teasers_only)
