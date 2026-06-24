"""QA round 5 regression test — R5-con5 (S3): an empty/whitespace whitelist term
(a hand-edited config.yaml artifact) must not match every paper.

``_build_pattern("")`` produced ``r"\b\b"``, which matches at any word boundary,
so a blank whitelist entry yielded a false Author/Title match on *all* papers.
"""

from __future__ import annotations

import unittest

from app.services.matching import check_author_match, check_whitelist_match


class EmptyWhitelistTermTests(unittest.TestCase):
    def test_empty_term_does_not_match_everything(self):
        self.assertEqual(check_whitelist_match(["Some paper about vision"], [""]), [])

    def test_whitespace_term_does_not_match(self):
        self.assertEqual(check_whitelist_match(["Another paper title"], ["   "]), [])

    def test_real_term_still_matches(self):
        self.assertEqual(check_whitelist_match(["Deep Vision models"], ["Vision"]), ["Vision"])

    def test_empty_author_term_does_not_match(self):
        self.assertEqual(check_author_match(["Jane Doe", "John Smith"], [""]), [])


if __name__ == "__main__":
    unittest.main()
