"""PDF section extraction for arXiv papers using pdfplumber."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field

import pdfplumber

LOGGER = logging.getLogger(__name__)

SECTION_TYPES = [
    "abstract",
    "introduction",
    "related work",
    "method",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "acknowledgments",
    "references",
]

# Match numbered or unnumbered headings common in arXiv papers.
# Examples: "1. Introduction", "2 Method", "III. EXPERIMENTS", "Abstract", "CONCLUSION"
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:[0-9]+\.?\s+)|"  # "1. " or "1 "
    r"(?:[IVXLC]+\.?\s+)"  # "III. " or "IV "
    r")?"
    r"("
    + "|".join(re.escape(s) for s in SECTION_TYPES)
    + r")"
    r"(?:\s*[:.]?\s*$|\s+)",
    re.IGNORECASE,
)

# Broader heading pattern for detecting section boundaries by formatting.
_ALLCAPS_HEADING_RE = re.compile(
    r"^(?:[0-9]+\.?\s+|[IVXLC]+\.?\s+)?[A-Z][A-Z\s]{2,}$"
)


@dataclass
class ExtractedSection:
    """A section extracted from a PDF."""

    section_type: str
    text: str
    order_index: int


def _normalize_section_type(raw: str) -> str:
    """Map heading text to a canonical section_type."""
    lower = raw.strip().lower()
    if lower in ("methods", "methodology", "approach", "proposed method"):
        return "method"
    if lower in ("experimental results", "evaluation"):
        return "results"
    if lower in ("related works", "background", "prior work"):
        return "related work"
    if lower in ("conclusions", "summary", "concluding remarks"):
        return "conclusion"
    if lower in ("acknowledgements", "acknowledgment"):
        return "acknowledgments"
    for st in SECTION_TYPES:
        if st in lower:
            return st
    return lower


def _extract_full_text(pdf_content: bytes) -> str:
    """Extract all text from a PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_sections(pdf_content: bytes) -> list[ExtractedSection]:
    """Extract structured sections from a PDF.

    Uses regex heuristics to detect section headings in arXiv-style papers.
    Returns sections in document order.
    """
    full_text = _extract_full_text(pdf_content)
    if not full_text.strip():
        return []

    lines = full_text.split("\n")

    # Find section boundaries.
    boundaries: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Try known section heading regex.
        match = _HEADING_RE.match(stripped)
        if match:
            section_name = _normalize_section_type(match.group(1))
            boundaries.append((i, section_name))
            continue

        # Try all-caps heading detection for short lines (likely headings).
        if len(stripped) < 60 and _ALLCAPS_HEADING_RE.match(stripped):
            section_name = _normalize_section_type(stripped.split(None, 1)[-1] if stripped[0].isdigit() else stripped)
            if section_name in SECTION_TYPES:
                boundaries.append((i, section_name))

    if not boundaries:
        return []

    # Build sections from boundaries.
    sections = []
    for idx, (start_line, section_type) in enumerate(boundaries):
        end_line = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        # Skip the heading line itself.
        section_text = "\n".join(lines[start_line + 1 : end_line]).strip()
        if section_text:
            sections.append(
                ExtractedSection(
                    section_type=section_type,
                    text=section_text,
                    order_index=idx,
                )
            )

    return sections


def extract_and_store_sections(
    paper_id: int,
    pdf_content: bytes,
    app=None,
) -> int:
    """Extract sections from PDF and store as PaperSection rows.

    Returns count of sections stored.
    """
    from app.models import PaperSection, db

    sections = extract_sections(pdf_content)
    if not sections:
        return 0

    def _store():
        # Remove existing sections for this paper (re-extraction).
        PaperSection.query.filter_by(paper_id=paper_id).delete()
        for sec in sections:
            db.session.add(
                PaperSection(
                    paper_id=paper_id,
                    section_type=sec.section_type,
                    text=sec.text,
                    order_index=sec.order_index,
                )
            )
        db.session.commit()

    if app is not None:
        with app.app_context():
            _store()
    else:
        _store()

    return len(sections)
