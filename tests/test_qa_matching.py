"""QA tests for whitelist matching, muting, and match priority.

Covers: CVARX-55 (Matching & Filtering)
- Whitelist matching: author, affiliation, title
- Match priority ordering: Author > Affiliation > Title
- Compound queries with negation
- Case-sensitive short acronyms vs case-insensitive normal terms
- Multi-word affiliation matching with separator flexibility
- Muting: author, affiliation, topic filtering on dashboard
"""

from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from app.models import Paper, db
from app.services.matching import (
    MATCH_PRIORITY,
    check_author_match,
    check_whitelist_match,
)
from tests.helpers import FlaskDBTestCase


class WhitelistMatchTests(unittest.TestCase):
    """Unit tests for check_whitelist_match."""

    def test_simple_title_keyword_match(self):
        matches = check_whitelist_match(
            ["A Novel Vision Transformer for Segmentation"],
            ["Vision"],
        )
        self.assertEqual(matches, ["Vision"])

    def test_case_insensitive_normal_term(self):
        matches = check_whitelist_match(
            ["vision transformers are powerful"],
            ["Vision"],
        )
        self.assertEqual(matches, ["Vision"])

    def test_case_sensitive_short_acronym(self):
        matches = check_whitelist_match(
            ["We train a GAN-based model"],
            ["GAN"],
        )
        self.assertEqual(matches, ["GAN"])

    def test_case_sensitive_short_acronym_no_match_lowercase(self):
        matches = check_whitelist_match(
            ["the gan approach works well"],
            ["GAN"],
        )
        self.assertEqual(matches, [])

    def test_compound_query_positive_and_negative(self):
        matches = check_whitelist_match(
            ["Transformer model for image segmentation"],
            [["Transformer", "!LSTM"]],
        )
        self.assertEqual(len(matches), 1)
        self.assertIn("Transformer", matches[0])

    def test_compound_query_negative_blocks_match(self):
        matches = check_whitelist_match(
            ["Comparing Transformer and LSTM architectures"],
            [["Transformer", "!LSTM"]],
        )
        self.assertEqual(matches, [])

    def test_compound_query_all_positive_must_match(self):
        matches = check_whitelist_match(
            ["Vision Transformer for detection"],
            [["Vision", "Transformer"]],
        )
        self.assertEqual(len(matches), 1)

    def test_compound_query_partial_positive_no_match(self):
        matches = check_whitelist_match(
            ["Vision model for detection"],
            [["Vision", "Transformer"]],
        )
        self.assertEqual(matches, [])

    def test_multi_word_affiliation_with_space(self):
        matches = check_whitelist_match(
            ["Researchers from Carnegie Mellon University"],
            ["Carnegie Mellon"],
        )
        self.assertEqual(matches, ["Carnegie Mellon"])

    def test_multi_word_affiliation_with_hyphen(self):
        matches = check_whitelist_match(
            ["Researchers from Carnegie-Mellon University"],
            ["Carnegie Mellon"],
        )
        self.assertEqual(matches, ["Carnegie Mellon"])

    def test_no_match_returns_empty(self):
        matches = check_whitelist_match(
            ["Quantum computing advances"],
            ["Vision", "Transformer"],
        )
        self.assertEqual(matches, [])

    def test_multiple_matches_deduped(self):
        matches = check_whitelist_match(
            ["Vision Transformer", "Another Vision paper"],
            ["Vision"],
        )
        self.assertEqual(matches, ["Vision"])

    def test_empty_whitelist(self):
        matches = check_whitelist_match(["Some title"], [])
        self.assertEqual(matches, [])

    def test_empty_texts(self):
        matches = check_whitelist_match([], ["Vision"])
        self.assertEqual(matches, [])


class AuthorMatchTests(unittest.TestCase):
    """Unit tests for check_author_match."""

    def test_exact_author_match(self):
        matches = check_author_match(["Jane Doe", "John Smith"], ["Jane Doe"])
        self.assertEqual(matches, ["Jane Doe"])

    def test_case_insensitive_author_match(self):
        matches = check_author_match(["jane doe"], ["Jane Doe"])
        self.assertEqual(matches, ["Jane Doe"])

    def test_no_author_match(self):
        matches = check_author_match(["Alice Brown"], ["Jane Doe"])
        self.assertEqual(matches, [])

    def test_multiple_author_matches(self):
        matches = check_author_match(
            ["Jane Doe", "John Smith"],
            ["Jane Doe", "John Smith"],
        )
        self.assertIn("Jane Doe", matches)
        self.assertIn("John Smith", matches)

    def test_partial_name_no_match(self):
        matches = check_author_match(["Jane"], ["Jane Doe"])
        self.assertEqual(matches, [])


class MatchPriorityTests(unittest.TestCase):
    """Verify match priority: Author > Affiliation > Title."""

    def test_author_higher_priority_than_affiliation(self):
        self.assertLess(MATCH_PRIORITY["Author"], MATCH_PRIORITY["Affiliation"])

    def test_affiliation_higher_priority_than_title(self):
        self.assertLess(MATCH_PRIORITY["Affiliation"], MATCH_PRIORITY["Title"])


class MutingDashboardTests(FlaskDBTestCase):
    """Test that muted authors/affiliations/topics hide papers on the dashboard."""

    def setUp(self):
        super().setUp()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        today = date.today()

        self.app.config["SCRAPER_CONFIG"]["preferences"]["muted"] = {
            "authors": ["Muted Author"],
            "affiliations": [],
            "topics": ["Quantum"],
        }

        db.session.add(
            Paper(
                arxiv_id="2607.0001",
                title="Visible Paper on Vision",
                authors="Good Author",
                link="https://arxiv.org/abs/2607.0001",
                pdf_link="https://arxiv.org/pdf/2607.0001",
                abstract_text="about vision",
                summary_text="summary",
                topic_tags=["Vision"],
                categories=["cs.CV"],
                match_type="Title",
                matched_terms=["Vision"],
                paper_score=15.0,
                feedback_score=0,
                is_hidden=False,
                publication_date=today.isoformat(),
                publication_dt=today,
                scraped_date=today.isoformat(),
                scraped_at=now,
            )
        )
        db.session.add(
            Paper(
                arxiv_id="2607.0002",
                title="Paper by Muted Author",
                authors="Muted Author",
                link="https://arxiv.org/abs/2607.0002",
                pdf_link="https://arxiv.org/pdf/2607.0002",
                abstract_text="about stuff",
                summary_text="summary",
                topic_tags=["Vision"],
                categories=["cs.CV"],
                match_type="Author",
                matched_terms=["Muted Author"],
                paper_score=20.0,
                feedback_score=0,
                is_hidden=False,
                publication_date=today.isoformat(),
                publication_dt=today,
                scraped_date=today.isoformat(),
                scraped_at=now,
            )
        )
        db.session.add(
            Paper(
                arxiv_id="2607.0003",
                title="Quantum Computing Paper",
                authors="Good Author",
                link="https://arxiv.org/abs/2607.0003",
                pdf_link="https://arxiv.org/pdf/2607.0003",
                abstract_text="about quantum",
                summary_text="summary",
                topic_tags=["Quantum"],
                categories=["cs.CV"],
                match_type="Title",
                matched_terms=["Quantum"],
                paper_score=18.0,
                feedback_score=0,
                is_hidden=False,
                publication_date=today.isoformat(),
                publication_dt=today,
                scraped_date=today.isoformat(),
                scraped_at=now,
            )
        )
        db.session.commit()
        self.client = self.app.test_client()

    def test_muted_author_hidden_from_inbox(self):
        response = self.client.get("/?timeframe=all")
        text = response.get_data(as_text=True)
        self.assertIn("Visible Paper on Vision", text)
        self.assertNotIn("Paper by Muted Author", text)

    def test_muted_topic_hidden_from_inbox(self):
        response = self.client.get("/?timeframe=all")
        text = response.get_data(as_text=True)
        self.assertIn("Visible Paper on Vision", text)
        self.assertNotIn("Quantum Computing Paper", text)


if __name__ == "__main__":
    unittest.main()
