"""QA round 5 regression test — R5-con4 (S2): parse_venue must not steal an
acceptance/qualifier signal that belongs to a *co-mentioned* venue.

parse_venue reports the first venue in KNOWN_VENUES order, then weighs nearby
submit/accept/qualifier/workshop cues. Prior rounds gated those cues by proximity
*window* to the matched venue, but a cue inside that window can still be nearer to
a second, also-mentioned venue — so it was attributed to the wrong venue. The fix
only attributes a cue when the matched venue is the nearest venue mention to it.
"""

from __future__ import annotations

import unittest

from app.services.venues import parse_venue


class VenueSignalOwnershipTests(unittest.TestCase):
    def test_qualifier_near_other_venue_not_stolen(self):
        # CVPR is matched first; the "oral" qualifier belongs to ICCV (nearer it).
        vm = parse_venue("Accepted to CVPR 2024. Our ICCV oral paper is extended here.")
        assert vm is not None
        self.assertEqual(vm.venue, "CVPR")
        self.assertNotIn(vm.status, ("oral", "spotlight", "highlight"))
        self.assertEqual(vm.status, "accepted")

    def test_single_venue_oral_still_detected(self):
        vm = parse_venue("Accepted to CVPR 2024 (Oral).")
        assert vm is not None
        self.assertEqual(vm.venue, "CVPR")
        self.assertEqual(vm.status, "oral")

    def test_single_venue_submitted_still_mentioned(self):
        vm = parse_venue("Submitted to CVPR 2025.")
        assert vm is not None
        self.assertEqual(vm.venue, "CVPR")
        self.assertEqual(vm.status, "mentioned")


if __name__ == "__main__":
    unittest.main()
