"""Generate PDF thumbnails natively using pdfplumber."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import requests
import pdfplumber

from app.services.http_client import request_with_backoff

LOGGER = logging.getLogger(__name__)

def generate_thumbnail(
    arxiv_id: str,
    pdf_link: str,
    static_dir: str | Path,
    session: requests.Session | None = None,
) -> bool:
    """Download the PDF, generate a thumbnail of its first page, and save it."""
    thumbnails_dir = Path(static_dir) / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    out_path = thumbnails_dir / f"{arxiv_id}.png"
    if out_path.exists():
        return True

    try:
        response = request_with_backoff(
            "GET",
            pdf_link,
            timeout=30,
            attempts=2,
            base_delay=1.0,
            session=session,
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp.flush()

            with pdfplumber.open(tmp.name) as pdf:
                if not pdf.pages:
                    return False
                first_page = pdf.pages[0]
                # Render to image using resolution 72 (default is fine for small thumbnails)
                im = first_page.to_image(resolution=72)
                im.save(str(out_path), format="PNG")

        LOGGER.info("Successfully generated thumbnail for %s", arxiv_id)
        return True
    except Exception as exc:
        LOGGER.warning("Thumbnail generation failed for %s: %s", arxiv_id, exc)
        return False
