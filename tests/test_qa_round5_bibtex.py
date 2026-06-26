"""QA round 5 regression test — R5-con11 (S3): a BibTeX cite key derived from a
non-arXiv link must be sanitized to BibTeX-legal characters (and non-empty), or it
emits a malformed/un-citable @article entry.
"""

from __future__ import annotations

import re
import unittest
from types import SimpleNamespace

from app.services.bibtex import _make_cite_key

_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class CiteKeyTests(unittest.TestCase):
    def test_non_arxiv_link_is_sanitized(self):
        paper = SimpleNamespace(arxiv_id=None, link="https://example.com/papers/my paper?id=1&x=2", id=42)
        key = _make_cite_key(paper)
        self.assertRegex(key, _KEY_RE)

    def test_empty_link_falls_back_to_paper_id(self):
        paper = SimpleNamespace(arxiv_id=None, link="", id=42)
        self.assertEqual(_make_cite_key(paper), "paper_42")

    def test_arxiv_id_key_unchanged(self):
        paper = SimpleNamespace(arxiv_id="2301.00001", link="x", id=1)
        self.assertEqual(_make_cite_key(paper), "2301_00001")


if __name__ == "__main__":
    unittest.main()
