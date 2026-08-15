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


class GraphApiTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _seed(self):
        a = _paper("2601.00001", "W1", ["W2"])
        b = _paper("2601.00002", "W2")
        db.session.add_all([a, b])
        db.session.commit()
        sync_citation_edges()
        return a, b

    def test_graph_returns_nodes_edges_and_pagerank(self):
        a, b = self._seed()

        response = self.client.get("/api/graph")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual({n["id"] for n in data["nodes"]}, {a.id, b.id})
        self.assertEqual(data["edges"], [{"source": a.id, "target": b.id}])
        by_id = {n["id"]: n for n in data["nodes"]}
        self.assertGreater(by_id[b.id]["pagerank"], by_id[a.id]["pagerank"])  # b is cited
        for key in ("title", "year", "citations", "tags"):
            self.assertIn(key, by_id[a.id])

    def test_graph_collection_filter(self):
        from app.models import Collection, PaperCollection

        a, b = self._seed()
        collection = Collection(name="Subset")
        db.session.add(collection)
        db.session.flush()
        db.session.add(PaperCollection(paper_id=a.id, collection_id=collection.id))
        db.session.commit()

        data = self.client.get(f"/api/graph?collection={collection.id}").get_json()

        self.assertEqual([n["id"] for n in data["nodes"]], [a.id])
        self.assertEqual(data["edges"], [])  # cited paper filtered out

        self.assertEqual(self.client.get("/api/graph?collection=9999").status_code, 404)

    def test_graph_limit_validation(self):
        self.assertEqual(self.client.get("/api/graph?limit=0").status_code, 400)
        self.assertEqual(self.client.get("/api/graph?limit=99999").status_code, 400)

    def test_graph_page_renders(self):
        response = self.client.get("/graph")

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn('id="graph"', body)
        self.assertIn("vis-network.min.js", body)


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
