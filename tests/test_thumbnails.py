from unittest.mock import MagicMock, patch

from app.services.thumbnail_generator import generate_thumbnail


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

        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_image = MagicMock()
        mock_page.to_image.return_value = mock_image
        mock_pdf.pages = [mock_page]
        mock_open.return_value.__enter__.return_value = mock_pdf

        result = generate_thumbnail("1234.5678", "http://fake.pdf", static_dir)

        assert result is True
        mock_req.assert_called_once()
        mock_open.assert_called_once()
        mock_page.to_image.assert_called_once_with(resolution=72)

        thumbnails_dir = static_dir / "thumbnails"
        out_path = thumbnails_dir / "1234.5678.png"
        mock_image.save.assert_called_once_with(str(out_path), format="PNG")


def test_generate_thumbnail_file_exists(tmp_path):
    static_dir = tmp_path / "static"
    thumbnails_dir = static_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True)
    out_path = thumbnails_dir / "1234.5678.png"
    out_path.touch()

    with patch("app.services.thumbnail_generator.request_with_backoff") as mock_req:
        result = generate_thumbnail("1234.5678", "http://fake.pdf", static_dir)
        assert result is True
        mock_req.assert_not_called()


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

        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_image = MagicMock()
        mock_page.to_image.return_value = mock_image
        mock_pdf.pages = [mock_page]

        ctx = MagicMock()
        ctx.__enter__.return_value = mock_pdf
        ctx.__exit__.return_value = None
        mock_open.side_effect = [ValueError("bad cached pdf"), ctx]

        result = generate_thumbnail(
            "1234.5678",
            "http://fake.pdf",
            static_dir,
            pdf_content=b"%PDF-1.4 cached pdf content",
        )

        assert result is True
        mock_req.assert_called_once()
        assert mock_open.call_count == 2
