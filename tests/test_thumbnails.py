from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.thumbnail_generator import extract_teaser_image, generate_thumbnail


def _touching_save(path, **_kwargs):
    Path(path).touch()


def _mock_pdf_context(num_pages: int = 1):
    """A pdfplumber.open() context manager whose pages render via mocks."""
    mock_pdf = MagicMock()
    mock_image = MagicMock()
    mock_image.save.side_effect = _touching_save
    pages = []
    for _ in range(num_pages):
        page = MagicMock()
        page.images = []
        page.to_image.return_value = mock_image
        pages.append(page)
    mock_pdf.pages = pages
    ctx = MagicMock()
    ctx.__enter__.return_value = mock_pdf
    ctx.__exit__.return_value = None
    return ctx, pages, mock_image


def test_generate_thumbnail_success(tmp_path):
    static_dir = tmp_path / "static"

    with (
        patch("app.services.thumbnail_generator.request_with_backoff") as mock_req,
        patch("app.services.thumbnail_generator.pdfplumber.open") as mock_open,
    ):
        mock_response = MagicMock()
        mock_response.content = b"%PDF-1.4 fake pdf content"
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_req.return_value = mock_response

        ctx, pages, mock_image = _mock_pdf_context()
        mock_open.return_value = ctx

        result = generate_thumbnail("1234.5678", "http://fake.pdf", static_dir)

        assert result is True
        mock_req.assert_called_once()
        # Page-1 render (150 DPI default) plus the teaser fallback render.
        pages[0].to_image.assert_called_with(resolution=150)
        thumbnails_dir = static_dir / "thumbnails"
        assert (thumbnails_dir / "1234.5678.png").exists()
        assert (thumbnails_dir / "1234.5678_teaser.png").exists()


def test_generate_thumbnail_skips_when_both_files_exist(tmp_path):
    static_dir = tmp_path / "static"
    thumbnails_dir = static_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True)
    (thumbnails_dir / "1234.5678.png").touch()
    (thumbnails_dir / "1234.5678_teaser.png").touch()

    with patch("app.services.thumbnail_generator.request_with_backoff") as mock_req:
        result = generate_thumbnail("1234.5678", "http://fake.pdf", static_dir)
        assert result is True
        mock_req.assert_not_called()


def test_generate_thumbnail_backfills_missing_teaser(tmp_path):
    static_dir = tmp_path / "static"
    thumbnails_dir = static_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True)
    (thumbnails_dir / "1234.5678.png").touch()

    with patch("app.services.thumbnail_generator.pdfplumber.open") as mock_open:
        ctx, _pages, _mock_image = _mock_pdf_context()
        mock_open.return_value = ctx

        result = generate_thumbnail(
            "1234.5678",
            "http://fake.pdf",
            static_dir,
            pdf_content=b"%PDF-1.4 cached pdf content",
        )

        assert result is True
        assert (thumbnails_dir / "1234.5678_teaser.png").exists()


def test_generate_thumbnail_failure(tmp_path):
    static_dir = tmp_path / "static"

    with patch("app.services.thumbnail_generator.request_with_backoff") as mock_req:
        mock_req.side_effect = Exception("Network error")
        result = generate_thumbnail("1234.5678", "http://fake.pdf", static_dir)
        assert result is False


def test_generate_thumbnail_retries_with_fresh_download_when_pdf_content_is_invalid(tmp_path):
    static_dir = tmp_path / "static"

    with (
        patch("app.services.thumbnail_generator.request_with_backoff") as mock_req,
        patch("app.services.thumbnail_generator.pdfplumber.open") as mock_open,
    ):
        mock_response = MagicMock()
        mock_response.content = b"%PDF-1.4 fresh pdf content"
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_req.return_value = mock_response

        ctx, _pages, _mock_image = _mock_pdf_context()
        calls = {"n": 0}

        def open_side_effect(*_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("bad cached pdf")
            return ctx

        mock_open.side_effect = open_side_effect

        result = generate_thumbnail(
            "1234.5678",
            "http://fake.pdf",
            static_dir,
            pdf_content=b"%PDF-1.4 cached pdf content",
        )

        assert result is True
        mock_req.assert_called_once()
        assert (static_dir / "thumbnails" / "1234.5678.png").exists()
        assert (static_dir / "thumbnails" / "1234.5678_teaser.png").exists()


# ── Teaser extraction against real PDFs ──────────────────────────────────────


def _image_pdf(width_px: int, height_px: int) -> bytes:
    """A real one-page PDF whose page is a single embedded image (via Pillow)."""
    import io

    from PIL import Image

    image = Image.new("RGB", (width_px, height_px), color=(120, 40, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PDF")
    return buffer.getvalue()


def _text_pdf() -> bytes:
    import os
    import tempfile

    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 6, txt="A text-only first page with no figures.", ln=1)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf.output(tmp.name)
    try:
        return Path(tmp.name).read_bytes()
    finally:
        os.unlink(tmp.name)


def test_extract_teaser_picks_dominant_image(tmp_path):
    out_path = tmp_path / "teaser.png"

    assert extract_teaser_image(_image_pdf(600, 400), out_path) is True
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_extract_teaser_rejects_text_only_pdf(tmp_path):
    out_path = tmp_path / "teaser.png"

    assert extract_teaser_image(_text_pdf(), out_path) is False
    assert not out_path.exists()


def test_extract_teaser_handles_invalid_bytes(tmp_path):
    out_path = tmp_path / "teaser.png"

    assert extract_teaser_image(b"not a pdf", out_path) is False
    assert not out_path.exists()


def test_thumbnail_warmer_dedupes_in_flight_keys(monkeypatch):
    """A burst of requests for the same paper triggers only one generation while
    the first is still running (so lazy <img> floods don't fan out into N renders)."""
    import threading

    from app.services import thumbnail_warmer as tw

    calls: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def fake_generate(storage_key, pdf_link, static_dir):
        calls.append(storage_key)
        started.set()
        release.wait(timeout=2)
        return True

    monkeypatch.setattr(tw, "generate_thumbnail", fake_generate)
    warmer = tw.ThumbnailWarmer(max_workers=1)
    try:
        warmer.warm("1234.5678", "http://example/pdf", "/tmp")
        assert started.wait(timeout=2)  # first job is running, key is in-flight
        warmer.warm("1234.5678", "http://example/pdf", "/tmp")  # deduped
        release.set()
        warmer._executor.shutdown(wait=True)
    finally:
        release.set()

    assert calls == ["1234.5678"]
