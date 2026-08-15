from __future__ import annotations

import unittest

from app.models import Paper, PaperRelation, db
from app.services.citation_graph import pagerank, sync_citation_edges
from tests.helpers import FlaskDBTestCase


def _paper(arxiv_id: str, openalex_id: str | None = None, referenced_works: list[str] | None = None) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        authors="Author A",
        link=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_link=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        match_type="title",
        matched_terms=["Vision"],
        paper_score=1.0,
        publication_date="2026-01-01",
        scraped_date="2026-01-01",
        openalex_id=openalex_id,
        referenced_works=referenced_works or [],
    )


class SyncCitationEdgesTests(FlaskDBTestCase):
    def test_creates_edges_only_for_known_papers(self):
        a = _paper("2601.00001", "W1", ["W2", "W999"])  # W999 not in DB
        b = _paper("2601.00002", "W2")
        db.session.add_all([a, b])
        db.session.commit()

        inserted = sync_citation_edges()

        self.assertEqual(inserted, 1)
        edge = PaperRelation.query.filter_by(relation_type="cites").one()
        self.assertEqual((edge.paper_id, edge.related_paper_id), (a.id, b.id))

    def test_idempotent_and_skips_self_cites(self):
        a = _paper("2601.00001", "W1", ["W1", "W2"])  # self-reference must be dropped
        b = _paper("2601.00002", "W2")
        db.session.add_all([a, b])
        db.session.commit()

        self.assertEqual(sync_citation_edges(), 1)
        self.assertEqual(sync_citation_edges(), 0)
        self.assertEqual(PaperRelation.query.filter_by(relation_type="cites").count(), 1)

    def test_picks_up_late_arriving_cited_paper(self):
        a = _paper("2601.00001", "W1", ["W2"])
        db.session.add(a)
        db.session.commit()
        self.assertEqual(sync_citation_edges(), 0)

        db.session.add(_paper("2601.00002", "W2"))
        db.session.commit()
        self.assertEqual(sync_citation_edges(), 1)


class PagerankTests(unittest.TestCase):
    def test_chain_ranks_most_cited_highest(self):
        # 1 cites 2, 2 cites 3: rank flows down the chain.
        ranks = pagerank([1, 2, 3], [(1, 2), (2, 3)])
        self.assertGreater(ranks[3], ranks[2])
        self.assertGreater(ranks[2], ranks[1])
        self.assertAlmostEqual(sum(ranks.values()), 1.0, places=5)

    def test_empty_and_edgeless_graphs(self):
        self.assertEqual(pagerank([], []), {})
        ranks = pagerank([7, 8], [])
        self.assertAlmostEqual(ranks[7], 0.5)
        self.assertAlmostEqual(ranks[8], 0.5)

    def test_edges_to_unknown_nodes_ignored(self):
        ranks = pagerank([1, 2], [(1, 2), (1, 99), (99, 2)])
        self.assertGreater(ranks[2], ranks[1])
