"""Tests for venue/acceptance detection from arXiv comments."""

from __future__ import annotations

import unittest

from app.services.venues import parse_venue, venue_bonus


class ParseVenueTests(unittest.TestCase):
    def test_accepted_to_cvpr_with_year(self):
        match = parse_venue("Accepted to CVPR 2026")
        self.assertIsNotNone(match)
        self.assertEqual(match.venue, "CVPR")
        self.assertEqual(match.year, 2026)
        self.assertEqual(match.status, "accepted")

    def test_oral_qualifier(self):
        match = parse_venue("ICCV 2025 oral presentation")
        self.assertEqual(match.venue, "ICCV")
        self.assertEqual(match.year, 2025)
        self.assertEqual(match.status, "oral")

    def test_spotlight_and_highlight_qualifiers(self):
        self.assertEqual(parse_venue("NeurIPS 2025 spotlight").status, "spotlight")
        self.assertEqual(parse_venue("CVPR 2026, selected as a highlight").status, "highlight")

    def test_to_appear_in_counts_as_accepted(self):
        match = parse_venue("To appear in NeurIPS")
        self.assertEqual(match.venue, "NeurIPS")
        self.assertIsNone(match.year)
        self.assertEqual(match.status, "accepted")

    def test_camera_ready_counts_as_accepted(self):
        self.assertEqual(parse_venue("Camera-ready version for ECCV 2026").status, "accepted")

    def test_submitted_is_only_mentioned(self):
        match = parse_venue("Submitted to ICLR 2026")
        self.assertEqual(match.venue, "ICLR")
        self.assertEqual(match.status, "mentioned")

    def test_under_review_is_only_mentioned(self):
        self.assertEqual(parse_venue("Under review at TPAMI").status, "mentioned")

    def test_workshop_status(self):
        match = parse_venue("Accepted at the CVPR 2026 Workshop on Autonomous Driving")
        self.assertEqual(match.venue, "CVPR")
        self.assertEqual(match.status, "workshop")

    def test_main_conference_oral_not_demoted_by_distant_workshop_mention(self):
        # The "workshop" token belongs to a different venue mentioned later; it
        # must not demote a genuine CVPR oral acceptance to "workshop".
        match = parse_venue("Accepted to CVPR 2024 (Oral). Extended version of our ICCV workshop paper.")
        self.assertEqual(match.venue, "CVPR")
        self.assertEqual(match.status, "oral")

    def test_submitted_with_distant_accepted_word_is_only_mentioned(self):
        # A stray, far-away "accepted" must not promote a submission to accepted.
        match = parse_venue("Submitted to CVPR; we hope it gets accepted.")
        self.assertEqual(match.venue, "CVPR")
        self.assertEqual(match.status, "mentioned")

    def test_nips_alias_maps_to_neurips(self):
        self.assertEqual(parse_venue("Published at NIPS 2017").venue, "NeurIPS")

    def test_year_adjacent_without_space(self):
        match = parse_venue("Accepted by CVPR2026")
        self.assertEqual(match.venue, "CVPR")
        self.assertEqual(match.year, 2026)

    def test_lowercase_rss_feed_does_not_match_rss_conference(self):
        self.assertIsNone(parse_venue("Fetched from the arXiv rss feed"))

    def test_uppercase_rss_conference_matches(self):
        match = parse_venue("Accepted to RSS 2026")
        self.assertEqual(match.venue, "RSS")
        self.assertEqual(match.status, "accepted")

    def test_plain_page_count_comment_has_no_venue(self):
        self.assertIsNone(parse_venue("10 pages, 5 figures"))

    def test_empty_or_none_comment(self):
        self.assertIsNone(parse_venue(None))
        self.assertIsNone(parse_venue(""))
        self.assertIsNone(parse_venue("   "))

    def test_siggraph_asia_preferred_over_siggraph(self):
        self.assertEqual(parse_venue("Accepted to SIGGRAPH Asia 2025").venue, "SIGGRAPH Asia")

    def test_venue_not_matched_inside_words(self):
        # "MICCAI" contains "ICCA"? guard: aliases must not match inside other words.
        self.assertIsNone(parse_venue("The discovery of NIPSomething"))

    def test_journal_long_form_alias(self):
        match = parse_venue("Published in IEEE Transactions on Pattern Analysis and Machine Intelligence")
        self.assertEqual(match.venue, "TPAMI")
        self.assertEqual(match.status, "accepted")


class VenueBonusTests(unittest.TestCase):
    def test_multipliers(self):
        self.assertEqual(venue_bonus("accepted", 8.0), 8.0)
        self.assertEqual(venue_bonus("oral", 8.0), 12.0)
        self.assertEqual(venue_bonus("spotlight", 8.0), 12.0)
        self.assertEqual(venue_bonus("highlight", 8.0), 12.0)
        self.assertEqual(venue_bonus("workshop", 8.0), 4.0)

    def test_mentioned_and_missing_score_zero(self):
        self.assertEqual(venue_bonus("mentioned", 8.0), 0.0)
        self.assertEqual(venue_bonus(None, 8.0), 0.0)
        self.assertEqual(venue_bonus("unknown-status", 8.0), 0.0)


if __name__ == "__main__":
    unittest.main()
