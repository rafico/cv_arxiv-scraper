"""QA round 4 regression tests for venue/acceptance detection.

G12: ``_nearest_distance`` measured the distance to the venue's *start* only,
so for long-form aliases a genuine acceptance phrase that *follows* the venue
name sat far from ``venue.start()`` and a preceding submission word was treated
as closer, demoting a real acceptance to ``mentioned``.
"""

from __future__ import annotations

import unittest

from app.services.venues import parse_venue


class QARound4VenueDistanceTests(unittest.TestCase):
    def test_g12_long_form_alias_trailing_acceptance_outranks_preceding_submission(self):
        # "under review" precedes the venue; "camera-ready" immediately follows
        # the long-form alias. Edge-aware distance makes the acceptance closer.
        match = parse_venue(
            "Extended version, under review elsewhere. International Journal of Computer Vision, camera-ready."
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.venue, "IJCV")
        self.assertEqual(match.status, "accepted")

    def test_g12_genuine_submissions_stay_mentioned(self):
        # No false promotion: bare submission cues must remain "mentioned".
        self.assertEqual(
            parse_venue("Submitted to CVPR; we hope it gets accepted.").status,
            "mentioned",
        )
        self.assertEqual(parse_venue("Submitted to ICLR 2026").status, "mentioned")
        self.assertEqual(parse_venue("Under review at TPAMI").status, "mentioned")


if __name__ == "__main__":
    unittest.main()
