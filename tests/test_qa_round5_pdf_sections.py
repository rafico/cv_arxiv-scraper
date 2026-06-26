"""QA round 5 regression test — R5-con12 (S2): a wrapped body line that merely
*starts* with a section keyword ("Results show our approach…") must not be treated
as a section heading, fabricating a boundary and truncating the real section.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.services.pdf_extraction import extract_sections

_LONG_BODY = (
    "Results show that our approach is state-of-the-art across many challenging "
    "benchmarks and experimental settings considered in this work."
)


class HeadingFalsePositiveTests(unittest.TestCase):
    @patch("app.services.pdf_extraction._extract_full_text")
    def test_body_line_starting_with_keyword_is_not_a_heading(self, mock_text):
        mock_text.return_value = "Introduction\nThis is the introduction.\n" + _LONG_BODY + "\nMore intro text.\n"
        sections = extract_sections(b"%PDF-fake")
        types = [s.section_type for s in sections]
        # The long "Results show…" body line must not open a 'results' section.
        self.assertNotIn("results", types)
        # The real "Introduction" heading is still detected, and its body keeps the
        # wrapped line that begins with "Results".
        self.assertIn("introduction", types)
        intro = next(s for s in sections if s.section_type == "introduction")
        self.assertIn("Results show", intro.text)

    @patch("app.services.pdf_extraction._extract_full_text")
    def test_short_real_heading_still_detected(self, mock_text):
        mock_text.return_value = "Introduction\nbody a\nResults\nbody b\n"
        sections = extract_sections(b"%PDF-fake")
        types = [s.section_type for s in sections]
        self.assertIn("results", types)


if __name__ == "__main__":
    unittest.main()
