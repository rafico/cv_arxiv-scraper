"""Generate PDF thumbnails natively using pdfplumber."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pdfplumber
import requests

from app.services.http_client import request_with_backoff

LOGGER = logging.getLogger(__name__)


def _looks_like_pdf(content: bytes | None) -> bool:
    if not content:
        return False
    return content.lstrip().startswith(b"%PDF-")


def _download_pdf(pdf_link: str, session: requests.Session | None = None) -> bytes:
    response = request_with_backoff(
        "GET",
        pdf_link,
        timeout=45,
        attempts=3,
        base_delay=1.5,
        headers={"Accept": "application/pdf"},
        session=session,
    )
    content = response.content
    if not _looks_like_pdf(content):
        content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0] or "unknown"
        raise ValueError(f"Response was not a PDF (content-type: {content_type})")
    return content


def _render_thumbnail(pdf_content: bytes, out_path: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(pdf_content)
        tmp.flush()

        with pdfplumber.open(tmp.name) as pdf:
            if not pdf.pages:
                raise ValueError("PDF had no pages")
            first_page = pdf.pages[0]
            im = first_page.to_image(resolution=72)
            im.save(str(out_path), format="PNG")

            if hasattr(im.original, "close"):
                im.original.close()


def generate_thumbnail(
    arxiv_id: str,
    pdf_link: str,
    static_dir: str | Path,
    session: requests.Session | None = None,
    pdf_content: bytes | None = None,
) -> bool:
    """Download the PDF, generate a thumbnail of its first page, and save it."""
    thumbnails_dir = Path(static_dir) / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    out_path = thumbnails_dir / f"{arxiv_id}.png"
    if out_path.exists():
        return True

    try:
        if pdf_content is not None:
            try:
                if not _looks_like_pdf(pdf_content):
                    raise ValueError("Provided PDF bytes were not a valid PDF")
                _render_thumbnail(pdf_content, out_path)
                LOGGER.info("Successfully generated thumbnail for %s", arxiv_id)
                return True
            except Exception as exc:
                LOGGER.debug("Retrying thumbnail generation for %s with a fresh PDF download: %s", arxiv_id, exc)

        content_to_use = _download_pdf(pdf_link, session=session)
        _render_thumbnail(content_to_use, out_path)
        LOGGER.info("Successfully generated thumbnail for %s", arxiv_id)
        return True
    except Exception as exc:
        LOGGER.warning("Thumbnail generation failed for %s: %s", arxiv_id, exc)
        return False
