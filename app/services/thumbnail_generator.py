"""Generate PDF thumbnails natively using pdfplumber."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pdfplumber
import requests

from app.services.http_client import request_with_backoff

LOGGER = logging.getLogger(__name__)

DEFAULT_THUMBNAIL_DPI = 150

# A teaser figure must dominate the page: at least 15% of the page area,
# reasonably wide, and not a thin rule or sidebar logo.
_TEASER_MIN_PAGE_AREA_RATIO = 0.15
_TEASER_MIN_WIDTH_PT = 200.0
_TEASER_ASPECT_RANGE = (0.2, 5.0)


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


def _render_thumbnail(pdf_content: bytes, out_path: Path, resolution: int = DEFAULT_THUMBNAIL_DPI) -> None:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(pdf_content)
        tmp.flush()

        with pdfplumber.open(tmp.name) as pdf:
            if not pdf.pages:
                raise ValueError("PDF had no pages")
            first_page = pdf.pages[0]
            im = first_page.to_image(resolution=resolution)
            im.save(str(out_path), format="PNG")

            if hasattr(im.original, "close"):
                im.original.close()


def _best_teaser_bbox(page) -> tuple[float, float, float, float] | None:
    """Largest embedded image on the page passing the teaser sanity filters."""
    page_area = float(page.width) * float(page.height)
    best_area = 0.0
    best_bbox = None

    for image in page.images:
        # Clamp to the page bounds: pdfplumber raises on out-of-page crops.
        x0 = max(float(image["x0"]), float(page.bbox[0]))
        top = max(float(image["top"]), float(page.bbox[1]))
        x1 = min(float(image["x1"]), float(page.bbox[2]))
        bottom = min(float(image["bottom"]), float(page.bbox[3]))
        width = x1 - x0
        height = bottom - top
        if width <= 0 or height <= 0:
            continue

        area = width * height
        aspect = width / height
        if (
            width < _TEASER_MIN_WIDTH_PT
            or area < _TEASER_MIN_PAGE_AREA_RATIO * page_area
            or not (_TEASER_ASPECT_RANGE[0] <= aspect <= _TEASER_ASPECT_RANGE[1])
        ):
            continue
        if area > best_area:
            best_area = area
            best_bbox = (x0, top, x1, bottom)

    return best_bbox


def extract_teaser_image(pdf_content: bytes, out_path: Path, resolution: int = DEFAULT_THUMBNAIL_DPI) -> bool:
    """Crop the teaser figure (largest qualifying image on pages 1-2) to a PNG.

    Returns False when no embedded image passes the filters (e.g. a text-only
    first page) — callers fall back to a full-page render.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(pdf_content)
            tmp.flush()

            with pdfplumber.open(tmp.name) as pdf:
                for page in pdf.pages[:2]:
                    bbox = _best_teaser_bbox(page)
                    if bbox is None:
                        continue
                    im = page.crop(bbox).to_image(resolution=resolution)
                    im.save(str(out_path), format="PNG")
                    if hasattr(im.original, "close"):
                        im.original.close()
                    return True
    except Exception as exc:
        LOGGER.debug("Teaser extraction failed: %s", exc)
    return False


def _render_teaser(pdf_content: bytes, out_path: Path, resolution: int = DEFAULT_THUMBNAIL_DPI) -> None:
    """Write the teaser figure, falling back to a page-1 render so the file
    always exists afterwards (keeps generation idempotent)."""
    if extract_teaser_image(pdf_content, out_path, resolution=resolution):
        return
    _render_thumbnail(pdf_content, out_path, resolution=resolution)


def _write_missing_renders(pdf_content: bytes, out_path: Path, teaser_path: Path, resolution: int) -> None:
    if not out_path.exists():
        _render_thumbnail(pdf_content, out_path, resolution=resolution)
    if not teaser_path.exists():
        _render_teaser(pdf_content, teaser_path, resolution=resolution)


def generate_thumbnail(
    arxiv_id: str,
    pdf_link: str,
    static_dir: str | Path,
    session: requests.Session | None = None,
    pdf_content: bytes | None = None,
    resolution: int = DEFAULT_THUMBNAIL_DPI,
) -> bool:
    """Download the PDF, then write the page-1 thumbnail and the teaser figure."""
    thumbnails_dir = Path(static_dir) / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    out_path = thumbnails_dir / f"{arxiv_id}.png"
    teaser_path = thumbnails_dir / f"{arxiv_id}_teaser.png"
    if out_path.exists() and teaser_path.exists():
        return True

    try:
        if pdf_content is not None:
            try:
                if not _looks_like_pdf(pdf_content):
                    raise ValueError("Provided PDF bytes were not a valid PDF")
                _write_missing_renders(pdf_content, out_path, teaser_path, resolution)
                LOGGER.info("Successfully generated thumbnail for %s", arxiv_id)
                return True
            except Exception as exc:
                LOGGER.debug("Retrying thumbnail generation for %s with a fresh PDF download: %s", arxiv_id, exc)

        content_to_use = _download_pdf(pdf_link, session=session)
        _write_missing_renders(content_to_use, out_path, teaser_path, resolution)
        LOGGER.info("Successfully generated thumbnail for %s", arxiv_id)
        return True
    except Exception as exc:
        LOGGER.warning("Thumbnail generation failed for %s: %s", arxiv_id, exc)
        return out_path.exists()
