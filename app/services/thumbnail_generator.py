"""Generate PDF thumbnails natively using pdfplumber."""

from __future__ import annotations

import io
import logging
import os
import uuid
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pdfplumber
import requests

from app.services.http_client import request_with_backoff
from app.services.subprocess_runner import run_isolated

LOGGER = logging.getLogger(__name__)

DEFAULT_THUMBNAIL_DPI = 150

# Wall-clock budget for rendering one PDF in an isolated process before giving up.
_RENDER_TIMEOUT = 120.0

# A teaser figure must dominate the page: at least 15% of the page area,
# reasonably wide, and not a thin rule or sidebar logo.
_TEASER_MIN_PAGE_AREA_RATIO = 0.15
_TEASER_MIN_WIDTH_PT = 200.0
_TEASER_ASPECT_RANGE = (0.2, 5.0)

# ── Inline figure previews ────────────────────────────────────────────────────
# Up to this many figures per paper, saved as {arxiv_id}_fig{n}.png alongside the
# page-1 thumbnail/teaser (deterministic names: no DB column needed).
MAX_PAPER_FIGURES = 4

# Each candidate <figure> image costs a download, so bound the per-paper fan-out;
# decorations (logos, footnote marks) are then dropped by the pixel filter.
_FIGURE_MAX_CANDIDATES = 8
# Below this decoded size an image is an icon or inline-math render, not a figure.
_FIGURE_MIN_WIDTH_PX = 240
_FIGURE_MIN_HEIGHT_PX = 120
# Byte ceilings for the arXiv HTML page and each figure asset (typically <1 MB).
_FIGURE_MAX_HTML_BYTES = 20 * 1024 * 1024
_FIGURE_MAX_IMAGE_BYTES = 15 * 1024 * 1024
# PDF-fallback filters (points, not pixels): looser than the teaser's page-dominance
# rule because legitimate later-page figures are smaller than a front-page teaser.
_PDF_FIGURE_MIN_WIDTH_PT = 150.0
_PDF_FIGURE_MIN_HEIGHT_PT = 80.0
_PDF_FIGURE_PAGES = 8
# Formats Pillow can decode into a PNG preview; skips .svg/.pdf vector assets.
_RASTER_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})


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
        # Mirror the affiliation prefetch's per-PDF ceiling: a "PDF" over 50 MB is
        # pathological and not worth buffering for a thumbnail.
        max_bytes=50 * 1024 * 1024,
    )
    content = response.content
    if not _looks_like_pdf(content):
        content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0] or "unknown"
        raise ValueError(f"Response was not a PDF (content-type: {content_type})")
    return content


def _save_image_atomic(im, out_path: Path) -> None:
    """Save a pdfplumber PageImage to ``out_path`` atomically.

    Render to a unique temp file in the SAME directory (so ``os.replace`` stays
    intra-filesystem and atomic on POSIX), then swap it onto the final path only
    after a successful save. A timeout (``proc.terminate``) or native Pillow crash
    mid-save then leaves the temp file (cleaned up here) rather than a truncated
    PNG at the served cache path.
    """
    tmp_path = out_path.with_name(f"{out_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        im.save(str(tmp_path), format="PNG")
        if hasattr(im.original, "close"):
            im.original.close()
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _render_thumbnail(pdf_content: bytes, out_path: Path, resolution: int = DEFAULT_THUMBNAIL_DPI) -> None:
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        if not pdf.pages:
            raise ValueError("PDF had no pages")
        first_page = pdf.pages[0]
        im = first_page.to_image(resolution=resolution)
        _save_image_atomic(im, out_path)


def _clamped_image_bbox(page, image) -> tuple[float, float, float, float] | None:
    """Image bbox clamped to the page bounds (pdfplumber raises on out-of-page
    crops), or None when the clamped box is degenerate."""
    x0 = max(float(image["x0"]), float(page.bbox[0]))
    top = max(float(image["top"]), float(page.bbox[1]))
    x1 = min(float(image["x1"]), float(page.bbox[2]))
    bottom = min(float(image["bottom"]), float(page.bbox[3]))
    if x1 - x0 <= 0 or bottom - top <= 0:
        return None
    return (x0, top, x1, bottom)


def _best_teaser_bbox(page) -> tuple[float, float, float, float] | None:
    """Largest embedded image on the page passing the teaser sanity filters."""
    page_area = float(page.width) * float(page.height)
    best_area = 0.0
    best_bbox = None

    for image in page.images:
        bbox = _clamped_image_bbox(page, image)
        if bbox is None:
            continue
        x0, top, x1, bottom = bbox
        width = x1 - x0
        height = bottom - top

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
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages[:2]:
                bbox = _best_teaser_bbox(page)
                if bbox is None:
                    continue
                im = page.crop(bbox).to_image(resolution=resolution)
                _save_image_atomic(im, out_path)
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
    thumbnails_dir = (Path(static_dir) / "thumbnails").resolve()

    out_path = (thumbnails_dir / f"{arxiv_id}.png").resolve()
    teaser_path = (thumbnails_dir / f"{arxiv_id}_teaser.png").resolve()
    # Defense in depth: arxiv_id derives from a remote feed link, so an id like
    # "../../etc/pwn" would otherwise write a PNG outside static/thumbnails. Reject
    # any id whose resolved paths escape the thumbnails dir before mkdir/write —
    # mirrors the serving-side guard in routes/dashboard.py.
    if not (out_path.is_relative_to(thumbnails_dir) and teaser_path.is_relative_to(thumbnails_dir)):
        LOGGER.warning("Refusing thumbnail for unsafe arxiv_id %r (path escapes thumbnails dir)", arxiv_id)
        return False
    # Legacy slash-form ids (e.g. 'cs/9901001') nest under a subdir; parents=True
    # creates the thumbnails dir and any nested subdir. The teaser shares the parent.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and teaser_path.exists():
        return True

    try:
        if pdf_content is not None:
            try:
                if not _looks_like_pdf(pdf_content):
                    raise ValueError("Provided PDF bytes were not a valid PDF")
                # Render in a child process: a native crash in pdfplumber/Pillow then
                # fails this paper instead of taking down the whole server.
                run_isolated(
                    _write_missing_renders, pdf_content, out_path, teaser_path, resolution, timeout=_RENDER_TIMEOUT
                )
                LOGGER.info("Successfully generated thumbnail for %s", arxiv_id)
                return True
            except Exception as exc:
                LOGGER.debug("Retrying thumbnail generation for %s with a fresh PDF download: %s", arxiv_id, exc)

        content_to_use = _download_pdf(pdf_link, session=session)
        run_isolated(_write_missing_renders, content_to_use, out_path, teaser_path, resolution, timeout=_RENDER_TIMEOUT)
        LOGGER.info("Successfully generated thumbnail for %s", arxiv_id)
        return True
    except Exception as exc:
        LOGGER.warning("Thumbnail generation failed for %s: %s", arxiv_id, exc)
        return out_path.exists()


# ── Inline figure previews ────────────────────────────────────────────────────


class _FigureImageParser(HTMLParser):
    """Collect <img src> values that appear inside <figure> elements, in order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._figure_depth = 0
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "figure":
            self._figure_depth += 1
        elif tag == "img" and self._figure_depth > 0:
            src = dict(attrs).get("src")
            if src and src not in self.sources:
                self.sources.append(src)

    def handle_endtag(self, tag: str) -> None:
        if tag == "figure" and self._figure_depth > 0:
            self._figure_depth -= 1


def parse_figure_image_urls(html_text: str, base_url: str) -> list[str]:
    """Absolute URLs of raster <figure> images in an arXiv HTML page, in order."""
    parser = _FigureImageParser()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception as exc:  # malformed markup — keep whatever was collected
        LOGGER.debug("Figure HTML parse stopped early: %s", exc)

    # arXiv serves the page at /html/{id}v{n} (no trailing slash) with assets at
    # /html/{id}v{n}/x1.png, so the base must end in a slash for urljoin.
    base = base_url if base_url.endswith("/") else f"{base_url}/"
    urls: list[str] = []
    for src in parser.sources:
        resolved = urljoin(base, src)
        if not resolved.startswith(("http://", "https://")):
            continue  # data: URIs and other non-fetchable schemes
        suffix = Path(urlparse(resolved).path).suffix.lower()
        if suffix and suffix not in _RASTER_EXTENSIONS:
            continue
        urls.append(resolved)
    return urls


def _decode_and_save_figures(image_blobs: list[bytes], out_paths: list[str]) -> int:
    """Decode candidate image bytes, keep sufficiently large ones, save as PNGs.

    Runs in an isolated child (Pillow is a native-crash site). Blobs that fail to
    decode or fall below the minimum pixel size are skipped; returns the number of
    figure files written (filling ``out_paths`` in order).
    """
    from PIL import Image

    saved = 0
    for blob in image_blobs:
        if saved >= len(out_paths):
            break
        try:
            with Image.open(io.BytesIO(blob)) as image:
                image.load()
                if image.width < _FIGURE_MIN_WIDTH_PX or image.height < _FIGURE_MIN_HEIGHT_PX:
                    continue
                out_image = image.convert("RGB") if image.mode not in ("RGB", "RGBA", "L", "LA") else image
                out_path = Path(out_paths[saved])
                tmp_path = out_path.with_name(f"{out_path.name}.{uuid.uuid4().hex}.tmp")
                try:
                    out_image.save(tmp_path, format="PNG")
                    os.replace(tmp_path, out_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
                saved += 1
        except Exception as exc:
            LOGGER.debug("Skipping undecodable figure image: %s", exc)
    return saved


def _extract_figures_from_html(
    arxiv_id: str,
    fig_paths: list[Path],
    session: requests.Session | None = None,
) -> int:
    """Fetch the arXiv HTML rendition and save its first qualifying figures.

    Raises on a missing/failed HTML fetch (404 is normal for older papers) so the
    caller can fall back to the PDF; per-image download failures are skipped.
    """
    response = request_with_backoff(
        "GET",
        f"https://arxiv.org/html/{arxiv_id}",
        timeout=30,
        attempts=2,
        base_delay=1.5,
        session=session,
        max_bytes=_FIGURE_MAX_HTML_BYTES,
    )
    urls = parse_figure_image_urls(response.text, str(response.url))[:_FIGURE_MAX_CANDIDATES]

    blobs: list[bytes] = []
    for url in urls:
        try:
            image_response = request_with_backoff(
                "GET",
                url,
                timeout=30,
                attempts=2,
                base_delay=1.0,
                session=session,
                max_bytes=_FIGURE_MAX_IMAGE_BYTES,
            )
            blobs.append(image_response.content)
        except Exception as exc:
            LOGGER.debug("Figure image download failed for %s: %s", url, exc)
    if not blobs:
        return 0

    # Decode in an isolated child: a native Pillow crash on one hostile image must
    # not take down the single worker.
    return run_isolated(_decode_and_save_figures, blobs, [str(path) for path in fig_paths], timeout=_RENDER_TIMEOUT)


def extract_pdf_figures(pdf_content: bytes, out_paths: list[str], resolution: int = DEFAULT_THUMBNAIL_DPI) -> int:
    """Crop the first sufficiently large embedded images from the leading pages.

    PDF fallback for papers without an arXiv HTML rendition. Intended to run via
    ``run_isolated`` (pdfplumber/Pillow are native-crash sites). Returns the number
    of figure files written, filling ``out_paths`` in order.
    """
    saved = 0
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages[:_PDF_FIGURE_PAGES]:
            for image in page.images:
                if saved >= len(out_paths):
                    return saved
                bbox = _clamped_image_bbox(page, image)
                if bbox is None:
                    continue
                x0, top, x1, bottom = bbox
                width = x1 - x0
                height = bottom - top
                aspect = width / height
                if (
                    width < _PDF_FIGURE_MIN_WIDTH_PT
                    or height < _PDF_FIGURE_MIN_HEIGHT_PT
                    or not (_TEASER_ASPECT_RANGE[0] <= aspect <= _TEASER_ASPECT_RANGE[1])
                ):
                    continue
                im = page.crop(bbox).to_image(resolution=resolution)
                _save_image_atomic(im, Path(out_paths[saved]))
                saved += 1
    return saved


def figure_paths_for(arxiv_id: str, static_dir: str | Path) -> list[Path]:
    """Existing figure files for a paper, in figure order.

    Figures are discoverable purely by their deterministic filenames
    (``{arxiv_id}_fig{n}.png``) — no DB column. The resolve/is_relative_to guard
    mirrors generate_thumbnail so a hostile id can't probe outside the cache dir.
    """
    thumbnails_dir = (Path(static_dir) / "thumbnails").resolve()
    paths: list[Path] = []
    for index in range(1, MAX_PAPER_FIGURES + 1):
        candidate = (thumbnails_dir / f"{arxiv_id}_fig{index}.png").resolve()
        if not candidate.is_relative_to(thumbnails_dir):
            return []
        if candidate.exists():
            paths.append(candidate)
    return paths


def generate_paper_figures(
    arxiv_id: str,
    static_dir: str | Path,
    *,
    session: requests.Session | None = None,
    pdf_content: bytes | None = None,
    pdf_link: str | None = None,
    resolution: int = DEFAULT_THUMBNAIL_DPI,
) -> int:
    """Extract up to :data:`MAX_PAPER_FIGURES` inline figure previews for a paper.

    The arXiv HTML rendition is the primary source (native-resolution assets);
    papers without one (a 404 is normal for older submissions) fall back to
    cropping embedded images from the first PDF pages, reusing ``pdf_content``
    when the scrape pipeline already holds the bytes or downloading via
    ``pdf_link`` otherwise. Best-effort: never raises; returns the number of
    figure files present afterwards.
    """
    thumbnails_dir = (Path(static_dir) / "thumbnails").resolve()
    fig_paths = [(thumbnails_dir / f"{arxiv_id}_fig{index}.png").resolve() for index in range(1, MAX_PAPER_FIGURES + 1)]
    # Same defense-in-depth as generate_thumbnail: arxiv_id derives from a remote
    # feed, so reject any id whose resolved paths escape the thumbnails dir.
    if not all(path.is_relative_to(thumbnails_dir) for path in fig_paths):
        LOGGER.warning("Refusing figures for unsafe arxiv_id %r (path escapes thumbnails dir)", arxiv_id)
        return 0

    existing = [path for path in fig_paths if path.exists()]
    if existing:
        return len(existing)  # a previous run already extracted figures
    fig_paths[0].parent.mkdir(parents=True, exist_ok=True)

    try:
        saved = _extract_figures_from_html(arxiv_id, fig_paths, session=session)
        if saved:
            LOGGER.info("Extracted %d figure(s) for %s from arXiv HTML", saved, arxiv_id)
            return saved
    except Exception as exc:
        LOGGER.debug("arXiv HTML figure extraction unavailable for %s: %s", arxiv_id, exc)

    try:
        content = pdf_content if _looks_like_pdf(pdf_content) else None
        if content is None and pdf_link:
            content = _download_pdf(pdf_link, session=session)
        if content is None:
            return 0
        saved = run_isolated(
            extract_pdf_figures, content, [str(path) for path in fig_paths], resolution, timeout=_RENDER_TIMEOUT
        )
        if saved:
            LOGGER.info("Extracted %d figure(s) for %s from PDF", saved, arxiv_id)
        return saved
    except Exception as exc:
        LOGGER.warning("Figure extraction failed for %s: %s", arxiv_id, exc)
        return sum(1 for path in fig_paths if path.exists())
