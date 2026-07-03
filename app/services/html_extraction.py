"""arXiv-HTML-first full-text section extraction.

arXiv publishes a clean LaTeXML HTML rendition at ``https://arxiv.org/html/{id}``
for essentially all new TeX submissions. Its ``<section>``/``<hN>`` structure,
literal ``<a href>`` links, and MathML ``alttext`` are far better inputs for the
section FAISS index (per-paper chat/RAG) than pdfplumber + regex heading guesses.

This module parses that HTML with the stdlib :mod:`html.parser` (no bs4), maps
headings onto the SAME canonical section vocabulary as
:mod:`app.services.pdf_extraction` (imported, not duplicated), and exposes an
HTML-first orchestrator (:func:`extract_sections`) that falls back to the existing
PDF extractor when HTML is unavailable (404 on older non-TeX papers) or unparseable.
Every network/parse failure degrades to ``None``/``"none"`` — never raises.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

import requests

from app.services.http_client import request_with_backoff

# Reuse the canonical vocabulary + normalizer — do NOT duplicate it.
from app.services.pdf_extraction import SECTION_TYPES, _normalize_section_type

LOGGER = logging.getLogger(__name__)

# Byte ceilings mirror the arXiv-HTML figure fetch already in thumbnail_generator:
# the HTML page is generous, a fallback PDF over 50 MB is pathological.
_MAX_HTML_BYTES = 20 * 1024 * 1024
_MAX_PDF_BYTES = 50 * 1024 * 1024

# Wall-clock budget for the isolated pdfplumber fallback (matches the scrape stage cap).
_PDF_STAGE_TIMEOUT = 900.0

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
# Tags whose content must never become section text.
_SKIP_TAGS = frozenset({"script", "style", "head", "noscript"})
# Block-level tags: emit a separator so adjacent paragraphs/list items don't run together.
_SPACE_TAGS = frozenset({"p", "div", "br", "li", "tr", "section", "blockquote", "figcaption", "td"})

# Only surface code/project-style links as the bonus field (a references list is full
# of DOI/arXiv anchors that classify as plain "web" and are dropped here).
_CODE_LINK_TYPES = frozenset({"code", "dataset", "demo", "project"})
_MAX_CODE_LINKS = 20

# Strip a leading section number/roman-numeral/appendix-letter token ("1", "2.3.",
# "III.", "A ") so "1 Introduction" / "III. Experiments" normalize like the PDF path.
_LEADING_NUMBER_RE = re.compile(r"^\s*(?:[0-9]+(?:\.[0-9]+)*\.?|[IVXLC]+\.?|[A-Z]\.)\s+")
_WS_RE = re.compile(r"\s+")


@dataclass
class HtmlSections:
    """Canonical sections + bonus code links parsed from an arXiv HTML rendition."""

    sections: list[tuple[str, str, int]]  # (section_type, text, order_index)
    links: list[dict[str, str]] = field(default_factory=list)


@dataclass
class SectionExtraction:
    """Result of the HTML-first orchestrator.

    ``source`` records provenance (``"html"``/``"pdf"``/``"none"``) without any DB
    column so callers can log HTML-vs-PDF counts.
    """

    sections: list[tuple[str, str, int]]  # (section_type, text, order_index)
    source: str
    links: list[dict[str, str]] = field(default_factory=list)


def _normalize_heading(raw: str) -> str:
    """Map a heading line ("1 Introduction", "III. Experiments") to a section_type."""
    stripped = _LEADING_NUMBER_RE.sub("", raw, count=1)
    return _normalize_section_type(stripped)


class _ArxivHtmlParser(HTMLParser):
    """Walk arXiv LaTeXML HTML into canonical (section_type, text) blocks.

    Text accumulates under the LAST canonical heading seen (abstract/introduction/
    method/…); non-canonical subsection headings are kept as body content rather than
    starting a new block, so a "Method" section keeps its subsection prose. MathML is
    replaced by its ``alttext`` (the source LaTeX) instead of the noisy element tree.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._current_type: str | None = None
        self._buf: list[str] = []
        self._heading_depth = 0
        self._heading_buf: list[str] = []
        self._skip_depth = 0
        self._math_depth = 0
        self._raw: list[tuple[str, str]] = []
        self.hrefs: list[str] = []

    def _append_text(self, text: str) -> None:
        if self._heading_depth:
            self._heading_buf.append(text)
        else:
            self._buf.append(text)

    def _flush(self) -> None:
        if self._current_type is not None:
            text = _WS_RE.sub(" ", "".join(self._buf)).strip()
            if text:
                self._raw.append((self._current_type, text))
        self._buf = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = dict(attrs)
        if tag == "a":
            href = attrs_d.get("href")
            if href:
                self.hrefs.append(href)
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "math":
            alttext = attrs_d.get("alttext")
            if alttext:
                self._append_text(f" {alttext} ")
            self._math_depth += 1
            return
        if self._math_depth:
            return
        if tag in _HEADING_TAGS:
            self._heading_depth += 1
            if self._heading_depth == 1:
                self._heading_buf = []
            return
        if tag in _SPACE_TAGS:
            self._append_text(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Self-closing <br/>, <img/>: only spacing matters (and <a/> hrefs).
        self.handle_starttag(tag, attrs)
        if tag not in _SKIP_TAGS and tag not in _HEADING_TAGS and tag != "math":
            return
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "math":
            if self._math_depth:
                self._math_depth -= 1
            return
        if self._math_depth:
            return
        if tag in _HEADING_TAGS:
            if self._heading_depth:
                self._heading_depth -= 1
            if self._heading_depth == 0:
                heading_text = "".join(self._heading_buf).strip()
                self._heading_buf = []
                self._handle_heading(heading_text)

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._math_depth:
            return
        self._append_text(data)

    def _handle_heading(self, heading_text: str) -> None:
        if not heading_text:
            return
        normalized = _normalize_heading(heading_text)
        if normalized in SECTION_TYPES:
            self._flush()
            self._current_type = normalized
        else:
            # A subsection title ("3.1 Network Architecture") — keep as body content
            # of the enclosing canonical section rather than a new boundary.
            self._append_text(f" {heading_text} ")

    def sections(self) -> list[tuple[str, str, int]]:
        self._flush()
        return [(section_type, text, index) for index, (section_type, text) in enumerate(self._raw)]


def _code_links(hrefs: list[str]) -> list[dict[str, str]]:
    """Classify literal <a href> anchors, keeping only code/project-style links."""
    if not hrefs:
        return []
    # Reuse the public categorizer via extract_resource_links (dedups, strips punctuation).
    from app.services.enrichment import extract_resource_links

    links = extract_resource_links(" ".join(hrefs))
    out = [link for link in links if link.get("type") in _CODE_LINK_TYPES]
    return out[:_MAX_CODE_LINKS]


def extract_html_sections(
    arxiv_id: str,
    *,
    session: requests.Session | None = None,
    scraper_config: dict | None = None,
    max_bytes: int = _MAX_HTML_BYTES,
) -> HtmlSections | None:
    """Fetch + parse the arXiv HTML rendition into canonical sections.

    Returns ``None`` on a clean miss — a 404 (older non-TeX paper), any fetch/parse
    failure, or an HTML page with no recognizable canonical sections — so the caller
    can fall back to the PDF path. Never raises.
    """
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        response = request_with_backoff(
            "GET",
            url,
            timeout=30,
            attempts=2,
            base_delay=1.5,
            session=session,
            scraper_config=scraper_config,
            max_bytes=max_bytes,
        )
    except Exception as exc:
        # 404 is normal for pre-HTML papers; anything else is a transient miss.
        LOGGER.debug("arXiv HTML fetch failed for %s: %s", arxiv_id, exc)
        return None

    parser = _ArxivHtmlParser()
    try:
        parser.feed(response.text)
        parser.close()
    except Exception as exc:
        LOGGER.debug("arXiv HTML parse failed for %s: %s", arxiv_id, exc)
        return None

    sections = parser.sections()
    if not sections:
        return None
    return HtmlSections(sections=sections, links=_code_links(parser.hrefs))


def _download_pdf(
    pdf_link: str,
    *,
    session: requests.Session | None = None,
    scraper_config: dict | None = None,
) -> bytes | None:
    """Best-effort PDF download for the fallback path (backfill has only a link)."""
    try:
        response = request_with_backoff(
            "GET",
            pdf_link,
            timeout=45,
            attempts=2,
            base_delay=1.5,
            headers={"Accept": "application/pdf"},
            session=session,
            scraper_config=scraper_config,
            max_bytes=_MAX_PDF_BYTES,
        )
    except Exception as exc:
        LOGGER.debug("PDF download failed for %s: %s", pdf_link, exc)
        return None
    content = response.content
    if not content or not content.lstrip().startswith(b"%PDF-"):
        return None
    return content


def _pdf_sections(pdf_content: bytes, *, isolate: bool = True) -> list[tuple[str, str, int]]:
    """Run the existing pdfplumber extractor, isolated by default (native-crash site)."""
    from app.services.pdf_extraction import extract_sections as extract_pdf_sections
    from app.services.pdf_extraction import extract_sections_batch

    if isolate:
        try:
            from app.services.subprocess_runner import run_isolated

            result = run_isolated(extract_sections_batch, [pdf_content], timeout=_PDF_STAGE_TIMEOUT)
            return list(result[0]) if result else []
        except Exception:
            LOGGER.warning("Isolated PDF section extraction failed (non-fatal)", exc_info=True)
            return []
    try:
        parsed = extract_pdf_sections(pdf_content)
    except Exception:
        LOGGER.warning("PDF section extraction failed (non-fatal)", exc_info=True)
        return []
    return [(sec.section_type, sec.text, sec.order_index) for sec in parsed]


def extract_sections(
    arxiv_id: str | None,
    *,
    pdf_content: bytes | None = None,
    pdf_link: str | None = None,
    session: requests.Session | None = None,
    scraper_config: dict | None = None,
    isolate_pdf: bool = True,
) -> SectionExtraction:
    """HTML-first section extraction with a PDF fallback.

    Tries the arXiv HTML rendition first; on a miss falls back to the existing
    pdfplumber path (downloading the PDF from ``pdf_link`` when no bytes are given).
    Always returns a :class:`SectionExtraction`; ``source`` records provenance.
    Never raises — every failure degrades to ``source="none"`` / empty sections.
    """
    if arxiv_id:
        html = extract_html_sections(arxiv_id, session=session, scraper_config=scraper_config)
        if html is not None and html.sections:
            return SectionExtraction(sections=html.sections, source="html", links=html.links)

    if pdf_content is None and pdf_link:
        pdf_content = _download_pdf(pdf_link, session=session, scraper_config=scraper_config)

    if pdf_content:
        sections = _pdf_sections(pdf_content, isolate=isolate_pdf)
        if sections:
            return SectionExtraction(sections=sections, source="pdf", links=[])

    return SectionExtraction(sections=[], source="none", links=[])
