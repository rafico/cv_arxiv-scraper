"""QA round 4 regression tests for app/services/thumbnail_generator.

G16: generate_thumbnail() must create the parent directory of the output path so
legacy slash-form arXiv ids (e.g. 'cs/9901001') write to a nested 'cs/' folder
instead of raising a swallowed FileNotFoundError.

G17: _render_thumbnail / extract_teaser_image must render atomically (temp file +
os.replace) so a crash/timeout mid-save never leaves a truncated PNG at the final
cache path that would then be served forever.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.thumbnail_generator import (
    _render_thumbnail,
    extract_teaser_image,
    generate_thumbnail,
)


def _touching_save(path, **_kwargs):
    Path(path).touch()


def _mock_pdf_context(num_pages: int = 1, save_side_effect=_touching_save):
    """A pdfplumber.open() context manager whose pages render via mocks."""
    mock_pdf = MagicMock()
    mock_image = MagicMock()
    mock_image.save.side_effect = save_side_effect
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


def test_generate_thumbnail_creates_nested_dir_for_legacy_id_g16(tmp_path):
    """G16: a legacy slash-form id writes the nested file rather than failing."""
    static_dir = tmp_path / "static"

    with patch("app.services.thumbnail_generator.pdfplumber.open") as mock_open:
        ctx, _pages, _mock_image = _mock_pdf_context()
        mock_open.return_value = ctx

        result = generate_thumbnail(
            "cs/9901001",
            "http://fake.pdf",
            static_dir,
            pdf_content=b"%PDF-1.4 cached pdf content",
        )

        assert result is True
        thumbnails_dir = static_dir / "thumbnails"
        assert (thumbnails_dir / "cs" / "9901001.png").exists()
        assert (thumbnails_dir / "cs" / "9901001_teaser.png").exists()


def test_render_thumbnail_leaves_no_partial_file_on_save_crash_g17(tmp_path):
    """G17: a crash mid-save must not leave a truncated PNG at the final path."""
    out_path = tmp_path / "1234.5678.png"

    def boom_save(path, **_kwargs):
        # Simulate Pillow writing some bytes then crashing mid-write.
        Path(path).write_bytes(b"\x89PNG truncated")
        raise RuntimeError("native crash mid-save")

    with patch("app.services.thumbnail_generator.pdfplumber.open") as mock_open:
        ctx, _pages, _mock_image = _mock_pdf_context(save_side_effect=boom_save)
        mock_open.return_value = ctx

        raised = False
        try:
            _render_thumbnail(b"%PDF-1.4 fake", out_path)
        except RuntimeError:
            raised = True

        assert raised
        # The destination must never hold the truncated render.
        assert not out_path.exists()
        # No leftover temp files in the directory either.
        assert list(tmp_path.iterdir()) == []


def test_extract_teaser_leaves_no_partial_file_on_save_crash_g17(tmp_path):
    """G17: teaser extraction must not leave a truncated PNG on a save crash."""
    out_path = tmp_path / "1234.5678_teaser.png"

    def boom_save(path, **_kwargs):
        Path(path).write_bytes(b"\x89PNG truncated")
        raise RuntimeError("native crash mid-save")

    mock_pdf = MagicMock()
    mock_image = MagicMock()
    mock_image.save.side_effect = boom_save
    page = MagicMock()
    crop_result = MagicMock()
    crop_result.to_image.return_value = mock_image
    page.crop.return_value = crop_result
    mock_pdf.pages = [page]
    ctx = MagicMock()
    ctx.__enter__.return_value = mock_pdf
    ctx.__exit__.return_value = None

    with (
        patch("app.services.thumbnail_generator.pdfplumber.open", return_value=ctx),
        patch(
            "app.services.thumbnail_generator._best_teaser_bbox",
            return_value=(0.0, 0.0, 100.0, 100.0),
        ),
    ):
        # extract_teaser_image swallows exceptions and returns False, but it must
        # not leave a partial file behind.
        result = extract_teaser_image(b"%PDF-1.4 fake", out_path)

    assert result is False
    assert not out_path.exists()
    assert list(tmp_path.iterdir()) == []
