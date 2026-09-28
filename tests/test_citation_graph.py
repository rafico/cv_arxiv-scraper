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

    def test_resolves_semantic_scholar_ids_too(self):
        a = _paper("2601.00001", None, ["abc123"])  # S2 hex id, no openalex ids anywhere
        b = _paper("2601.00002")
        b.semantic_scholar_id = "abc123"
        db.session.add_all([a, b])
        db.session.commit()

        self.assertEqual(sync_citation_edges(), 1)
        edge = PaperRelation.query.filter_by(relation_type="cites").one()
        self.assertEqual((edge.paper_id, edge.related_paper_id), (a.id, b.id))

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


def _s2(n: int) -> str:
    return f"{n:040x}"  # 40-char hex, the Semantic Scholar paperId shape


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class MissingReferencesTests(FlaskDBTestCase):
    def _members(self):
        from app.models import Collection, PaperCollection

        a = _paper("2601.00001", None, [_s2(1), _s2(2), _s2(3), _s2(9), "W77"])
        b = _paper("2601.00002", None, [_s2(1), _s2(2), _s2(9), "W77", _s2(1)])  # dup ref counts once
        c = _paper("2601.00003", None, [_s2(1), _s2(3)])
        local = _paper("2601.00009")
        local.semantic_scholar_id = _s2(9)  # already tracked: never a "prior work"
        collection = Collection(name="Review")
        db.session.add_all([a, b, c, local, collection])
        db.session.flush()
        db.session.add_all([PaperCollection(paper_id=p.id, collection_id=collection.id) for p in (a, b, c)])
        db.session.commit()
        return collection, [a.id, b.id, c.id]

    def test_ranks_outside_refs_cited_by_two_or_more_members(self):
        from app.services.citation_graph import missing_references

        _collection, ids = self._members()
        calls = []

        def fake_request(method, url, **kwargs):
            calls.append(kwargs)
            by_id = {
                _s2(1): {"title": "Deep Residual Learning", "year": 2016, "citationCount": 10, "externalIds": {}},
                _s2(2): {
                    "title": "Attention",
                    "year": 2017,
                    "citationCount": 500,
                    "externalIds": {"ArXiv": "1706.03762"},
                },
                _s2(3): {"title": "Adam", "year": 2014, "citationCount": 900, "externalIds": None},
            }
            return _FakeResponse([by_id[i] for i in kwargs["json"]["ids"]])

        result = missing_references(ids, request_fn=fake_request)

        self.assertEqual(len(calls), 1)  # one batch resolve
        self.assertEqual(calls[0]["attempts"], 1)
        self.assertEqual(sorted(calls[0]["json"]["ids"]), [_s2(1), _s2(2), _s2(3)])  # no W-ids, no local, no count-1
        self.assertNotIn("error", result)
        rows = result["results"]
        # (count, citationCount): 3 citers first, then the 2-citer ties by citationCount.
        self.assertEqual([r["s2_id"] for r in rows], [_s2(1), _s2(3), _s2(2)])
        self.assertEqual([r["cited_by"] for r in rows], [3, 2, 2])
        self.assertEqual(rows[2]["arxiv_id"], "1706.03762")
        self.assertEqual(rows[2]["link"], "https://arxiv.org/abs/1706.03762")
        self.assertEqual(rows[0]["link"], f"https://www.semanticscholar.org/paper/{_s2(1)}")
        self.assertEqual(rows[0]["year"], 2016)

        self.assertEqual(len(missing_references(ids, limit=1, request_fn=fake_request)["results"]), 1)

    def test_local_paper_without_s2_id_is_not_a_prior_work(self):
        from app.services.citation_graph import missing_references

        _collection, ids = self._members()
        db.session.add(_paper("1706.03762"))  # seeded by arXiv id: no semantic_scholar_id yet
        db.session.commit()

        def fake_request(method, url, **kwargs):
            return _FakeResponse(
                [
                    {"title": "T", "citationCount": 1, "externalIds": {"ArXiv": "1706.03762"} if i == _s2(2) else {}}
                    for i in kwargs["json"]["ids"]
                ]
            )

        rows = missing_references(ids, request_fn=fake_request)["results"]
        self.assertEqual(sorted(r["s2_id"] for r in rows), [_s2(1), _s2(3)])

    def test_no_candidates_skips_the_network(self):
        from app.services.citation_graph import missing_references

        a = _paper("2601.00001", None, [_s2(1)])
        db.session.add(a)
        db.session.commit()

        def boom(*args, **kwargs):
            raise AssertionError("no S2 call expected")

        self.assertEqual(missing_references([a.id], request_fn=boom), {"results": []})

    def test_s2_failure_degrades_to_error_payload(self):
        from app.services.citation_graph import missing_references

        _collection, ids = self._members()

        def down(*args, **kwargs):
            raise ConnectionError("429")

        self.assertEqual(
            missing_references(ids, request_fn=down), {"results": [], "error": "Semantic Scholar unavailable"}
        )

    def test_prior_works_route(self):
        from unittest.mock import patch

        client = self.app.test_client()
        collection, _ids = self._members()
        self.assertEqual(client.get("/api/collections/9999/prior-works").status_code, 404)

        with patch(
            "app.services.http_client.request_with_backoff",
            return_value=_FakeResponse([{"title": "T", "year": 2016, "citationCount": 1, "externalIds": {}}] * 3),
        ):
            response = client.get(f"/api/collections/{collection.id}/prior-works")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["paper_count"], 3)
        self.assertEqual(len(data["results"]), 3)

        with patch("app.services.http_client.request_with_backoff", side_effect=ConnectionError("down")):
            response = client.get(f"/api/collections/{collection.id}/prior-works")
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json()["error"], "Semantic Scholar unavailable")
