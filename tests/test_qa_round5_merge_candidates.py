"""QA round 5 regression test — R5-2 (S3): IngestOrchestrator._merge_candidates
must not collapse distinct link-less candidates under the empty-string key.

A non-arXiv feed item can have ``arxiv_id=None`` and ``link=""``; the old dedup
key ``candidate.arxiv_id or candidate.link`` evaluated to ``""`` for every such
candidate, so they all overwrote one another in the merge dict and all but one
were silently dropped before save (a NULL-``arxiv_id`` paper is persistable —
the unique index on ``arxiv_id`` is partial: ``WHERE arxiv_id IS NOT NULL``).
"""

from __future__ import annotations

import unittest

from app.services.ingest.base import PaperCandidate
from app.services.ingest.orchestrator import IngestOrchestrator


class MergeCandidatesLinklessTests(unittest.TestCase):
    def test_distinct_link_less_candidates_not_collapsed(self):
        c1 = PaperCandidate(arxiv_id=None, link="", title="Paper One")
        c2 = PaperCandidate(arxiv_id=None, link="", title="Paper Two")

        merged = IngestOrchestrator._merge_candidates([c1, c2], [])

        self.assertEqual(len(merged), 2)
        self.assertEqual({c.title for c in merged}, {"Paper One", "Paper Two"})

    def test_arxiv_id_dedup_still_applies(self):
        # A shared arxiv_id across primary/secondary still dedups (primary wins).
        primary = PaperCandidate(arxiv_id="2607.0001", link="https://arxiv.org/abs/2607.0001", title="Primary")
        secondary = PaperCandidate(arxiv_id="2607.0001", link="https://arxiv.org/abs/2607.0001", title="Secondary")

        merged = IngestOrchestrator._merge_candidates([primary], [secondary])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].title, "Primary")


if __name__ == "__main__":
    unittest.main()
