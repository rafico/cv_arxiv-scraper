"""Tests for PDF section extraction service."""

from __future__ import annotations

from app.services.pdf_extraction import (
    ExtractedSection,
    _normalize_section_type,
    extract_sections,
)


class TestNormalizeSectionType:
    def test_method_variants(self):
        assert _normalize_section_type("Methods") == "method"
        assert _normalize_section_type("Methodology") == "method"
        assert _normalize_section_type("Approach") == "method"
        assert _normalize_section_type("Proposed Method") == "method"

    def test_results_variants(self):
        assert _normalize_section_type("Experimental Results") == "results"
        assert _normalize_section_type("Evaluation") == "results"

    def test_conclusion_variants(self):
        assert _normalize_section_type("Conclusions") == "conclusion"
        assert _normalize_section_type("Summary") == "conclusion"

    def test_related_work_variants(self):
        assert _normalize_section_type("Related Works") == "related work"
        assert _normalize_section_type("Background") == "related work"

    def test_exact_types(self):
        assert _normalize_section_type("abstract") == "abstract"
        assert _normalize_section_type("introduction") == "introduction"
        assert _normalize_section_type("references") == "references"


def _make_pdf(lines: list[str]) -> bytes:
    """Create a minimal PDF with the given text lines using fpdf."""
    import tempfile

    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in lines:
        pdf.cell(0, 6, txt=line, ln=1)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf.output(tmp.name)
        tmp.seek(0)
        return open(tmp.name, "rb").read()


class TestExtractSections:
    def test_empty_pdf(self):
        result = extract_sections(_make_pdf([]))
        assert result == []

    def test_structured_paper(self):
        """Test with a PDF containing recognizable section headings."""
        lines = [
            "A Great Paper Title",
            "",
            "Abstract",
            "This paper presents a novel approach to widget detection.",
            "We show significant improvements over baselines.",
            "",
            "1. Introduction",
            "Object detection is an important problem in computer vision.",
            "Many approaches have been proposed in recent years.",
            "",
            "2. Method",
            "Our method uses a transformer backbone with custom attention.",
            "The architecture consists of three main components.",
            "",
            "3. Experiments",
            "We evaluate on COCO and Pascal VOC datasets.",
            "Results show state-of-the-art performance.",
            "",
            "4. Conclusion",
            "We presented a novel approach to widget detection.",
            "",
            "References",
            "[1] Some reference here.",
        ]
        sections = extract_sections(_make_pdf(lines))
        types = [s.section_type for s in sections]
        assert "abstract" in types
        assert "introduction" in types
        assert "method" in types
        assert "conclusion" in types

        # Verify sections are in increasing order.
        for i in range(1, len(sections)):
            assert sections[i].order_index > sections[i - 1].order_index

    def test_returns_extracted_section_dataclass(self):
        sec = ExtractedSection(section_type="abstract", text="Some text", order_index=0)
        assert sec.section_type == "abstract"
        assert sec.text == "Some text"
        assert sec.order_index == 0
