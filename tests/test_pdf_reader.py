from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.models import Paper, db
from tests.helpers import FlaskDBTestCase

PDF_BYTES = b"%PDF-1.4 fake pdf body"


def _paper() -> Paper:
    return Paper(
        arxiv_id="2601.00001",
        title="Readable Paper",
        authors="Author A",
        link="https://arxiv.org/abs/2601.00001",
        pdf_link="https://arxiv.org/pdf/2601.00001.pdf",
        match_type="title",
        matched_terms=["Vision"],
        paper_score=1.0,
        publication_date="2026-01-01",
        scraped_date="2026-01-01",
    )


class PdfReaderTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        self.paper = _paper()
        db.session.add(self.paper)
        db.session.commit()

    def _cache_path(self) -> Path:
        return Path(self.app.instance_path) / "pdfs" / f"{self.paper.id}.pdf"

    @patch("app.services.thumbnail_generator._download_pdf", return_value=PDF_BYTES)
    def test_first_request_downloads_and_caches(self, mock_download):
        response = self.client.get(f"/papers/{self.paper.id}/pdf")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/pdf")
        self.assertEqual(response.data, PDF_BYTES)
        self.assertEqual(self._cache_path().read_bytes(), PDF_BYTES)
        mock_download.assert_called_once_with(self.paper.pdf_link)

    @patch("app.services.thumbnail_generator._download_pdf", return_value=PDF_BYTES)
    def test_second_request_serves_from_cache(self, mock_download):
        self.client.get(f"/papers/{self.paper.id}/pdf")
        mock_download.reset_mock()

        response = self.client.get(f"/papers/{self.paper.id}/pdf")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, PDF_BYTES)
        mock_download.assert_not_called()

    @patch("app.services.thumbnail_generator._download_pdf", side_effect=ValueError("not a pdf"))
    def test_download_failure_returns_502(self, mock_download):
        response = self.client.get(f"/papers/{self.paper.id}/pdf")

        self.assertEqual(response.status_code, 502)
        self.assertFalse(self._cache_path().exists())

    def test_missing_paper_404s(self):
        self.assertEqual(self.client.get("/papers/9999/pdf").status_code, 404)
        self.assertEqual(self.client.get("/papers/9999/read").status_code, 404)

    def test_reader_page_renders(self):
        response = self.client.get(f"/papers/{self.paper.id}/read")

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn('id="pdf-viewer"', body)
        self.assertIn("Readable Paper", body)
        self.assertIn("pdfjs/pdf.min.mjs", body)
