"""Per-collection screening decisions (include / maybe / exclude; NULL = unscreened).

"exclude" drops a paper out of the review everywhere a collection is consumed,
while it stays a member so re-screening works and nothing re-suggests it.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

from sqlalchemy import inspect, text

from app.models import Collection, Paper, PaperCollection, PaperFeedback, db
from app.services import mcp_tools
from tests.helpers import FlaskDBTestCase


def _paper(idx: int, **overrides) -> Paper:
    today = date.today()
    defaults = dict(
        arxiv_id=f"2609.{5000 + idx:04d}",
        title=f"Screening Paper {idx}",
        authors="Author A",
        link=f"https://arxiv.org/abs/2609.{5000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2609.{5000 + idx:04d}",
        abstract_text="abstract",
        match_type="Title",
        is_hidden=False,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    defaults.update(overrides)
    return Paper(**defaults)


class ScreeningTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        # unscreened / include / maybe / exclude members, plus an outsider.
        self.papers = [_paper(i) for i in range(5)]
        self.collection = Collection(name="Review")
        db.session.add_all([*self.papers, self.collection])
        db.session.flush()
        for paper, decision in zip(self.papers[:4], (None, "include", "maybe", "exclude")):
            db.session.add(PaperCollection(paper_id=paper.id, collection_id=self.collection.id, decision=decision))
        db.session.commit()
        self.cid = self.collection.id
        self.unscreened, self.included, self.maybe, self.excluded, self.outsider = (p.id for p in self.papers)

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def _decision(self, paper_id: int):
        return PaperCollection.query.filter_by(paper_id=paper_id, collection_id=self.cid).one().decision

    def _view(self, query: str = "") -> str:
        return self.client.get(f"/?collection={self.cid}&timeframe=all{query}").get_data(as_text=True)

    @staticmethod
    def _card_ids(html: str) -> set[int]:
        return {int(pid) for pid in re.findall(r'data-paper-id="(\d+)"', html)}

    # ── schema ──

    def test_ensure_schema_adds_decision_to_legacy_db(self):
        from app.schema import ensure_schema

        db.session.execute(text("ALTER TABLE paper_collections DROP COLUMN decision"))
        db.session.commit()

        ensure_schema()
        ensure_schema()  # idempotent

        self.assertIn("decision", {c["name"] for c in inspect(db.engine).get_columns("paper_collections")})
        # Existing memberships survive and read as unscreened.
        self.assertEqual(PaperCollection.query.filter_by(collection_id=self.cid).count(), 4)
        self.assertIsNone(self._decision(self.included))

    # ── API ──

    def test_put_decision_sets_clears_and_validates(self):
        headers = {"X-CSRF-Token": self._csrf_token()}
        url = f"/api/collections/{self.cid}/papers/{self.unscreened}/decision"

        response = self.client.put(url, json={"decision": "include"}, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["decision"], "include")
        self.assertEqual(
            response.get_json()["counts"], {"all": 3, "unscreened": 0, "include": 2, "maybe": 1, "exclude": 1}
        )
        self.assertEqual(self._decision(self.unscreened), "include")

        self.assertEqual(self.client.put(url, json={"decision": None}, headers=headers).status_code, 200)
        self.assertIsNone(self._decision(self.unscreened))

        for bad in ({"decision": "bogus"}, {"decision": ["include"]}, {}):
            self.assertEqual(self.client.put(url, json=bad, headers=headers).status_code, 400, bad)
        outsider_url = f"/api/collections/{self.cid}/papers/{self.outsider}/decision"
        self.assertEqual(self.client.put(outsider_url, json={"decision": "include"}, headers=headers).status_code, 404)
        self.assertEqual(self.client.put(url, json={"decision": "include"}).status_code, 400)  # no CSRF
        # Review-scoped: never trains the ranker or hides the paper.
        self.assertEqual(PaperFeedback.query.count(), 0)
        self.assertFalse(db.session.get(Paper, self.unscreened).is_hidden)

    def test_bulk_decisions_touch_members_only(self):
        response = self.client.put(
            f"/api/collections/{self.cid}/decisions",
            json={"paper_ids": [self.unscreened, self.maybe, self.outsider, True], "decision": "exclude"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["updated"], 2)
        self.assertEqual(response.get_json()["counts"]["exclude"], 3)
        self.assertEqual(self._decision(self.maybe), "exclude")
        self.assertIsNone(PaperCollection.query.filter_by(paper_id=self.outsider).first())
        self.assertEqual(PaperFeedback.query.count(), 0)

    def test_add_as_not_relevant_is_an_excluded_member(self):
        headers = {"X-CSRF-Token": self._csrf_token()}
        url = f"/api/collections/{self.cid}/papers"
        self.assertEqual(
            self.client.post(url, json={"paper_id": self.outsider, "decision": "bogus"}, headers=headers).status_code,
            400,
        )
        response = self.client.post(url, json={"paper_id": self.outsider, "decision": "exclude"}, headers=headers)

        self.assertEqual(response.get_json()["added"], 1)
        self.assertEqual(self._decision(self.outsider), "exclude")

    # ── consumers: excluded is out of the review ──

    def test_collection_view_filters_by_decision_with_counts(self):
        default = self._view()
        self.assertEqual(self._card_ids(default), {self.unscreened, self.included, self.maybe})
        self.assertEqual(self._card_ids(self._view("&decision=unscreened")), {self.unscreened})
        self.assertEqual(self._card_ids(self._view("&decision=include")), {self.included})
        self.assertEqual(self._card_ids(self._view("&decision=exclude")), {self.excluded})

        self.assertIn('data-decision-count="all">3<', default)
        self.assertIn('data-decision-count="exclude">1<', default)
        self.assertRegex(default, r'data-decision-filter=""\s+data-active="true"')
        # Per-card control reflects the stored decision, plus a badge.
        self.assertRegex(default, r'data-decision="include"\s+data-active="true"')
        self.assertIn("data-decision-badge", default)
        self.assertIn('.decision-btn[data-decision="', default)  # i / e / m shortcuts
        self.assertIn("Not relevant", default)  # Suggest similar action

        self.client.put(
            f"/api/collections/{self.cid}/decisions",
            json={"paper_ids": [self.unscreened], "decision": "maybe"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertIn("All screened", self._view("&decision=unscreened"))

        inbox = self.client.get("/?timeframe=all").get_data(as_text=True)
        self.assertNotIn('class="decision-btn', inbox)
        self.assertNotIn("data-decision-filter=", inbox)

    def test_exports_and_graph_drop_excluded(self):
        bib = self.client.get(f"/api/export/bibtex?collection={self.cid}").get_data(as_text=True)
        self.assertIn("Screening Paper 1", bib)
        self.assertNotIn("Screening Paper 3", bib)

        nodes = self.client.get(f"/api/graph?collection={self.cid}").get_json()["nodes"]
        self.assertEqual({n["id"] for n in nodes}, {self.unscreened, self.included, self.maybe})

        counts = {c["id"]: c["paper_count"] for c in self.client.get("/api/collections").get_json()}
        self.assertEqual(counts[self.cid], 3)

    def test_csv_keeps_every_member_with_its_decision(self):
        rows = list(
            csv.DictReader(
                io.StringIO(self.client.get(f"/api/collections/{self.cid}/table.csv").get_data(as_text=True))
            )
        )
        self.assertEqual([r["decision"] for r in rows], ["", "include", "maybe", "exclude"])

    def test_prior_works_chat_and_neighbors_seed_from_the_review(self):
        with patch("app.services.citation_graph.missing_references", return_value={"results": []}) as refs:
            data = self.client.get(f"/api/collections/{self.cid}/prior-works").get_json()
        self.assertEqual(set(refs.call_args.args[0]), {self.unscreened, self.included, self.maybe})
        self.assertEqual(data["paper_count"], 3)

        with patch("app.services.rag.answer_query", return_value={}) as answer:
            self.client.post(
                "/api/corpus/chat",
                json={"query": "q", "collection_id": self.cid},
                headers={"X-CSRF-Token": self._csrf_token()},
            )
        self.assertEqual(set(answer.call_args.kwargs["paper_ids"]), {self.unscreened, self.included, self.maybe})

        with patch("app.services.corpus_analysis.find_neighbor_papers", return_value={"results": []}) as neighbors:
            self.client.get(f"/api/corpus/neighbors?collection_id={self.cid}")
        self.assertEqual(set(neighbors.call_args.args[0]), {self.unscreened, self.included, self.maybe})
        # Excluded members are no seed, but never come back as a suggestion either.
        self.assertIn(self.excluded, neighbors.call_args.kwargs["exclude_ids"])

    def test_find_neighbor_papers_skips_exclude_ids(self):
        from app.services.corpus_analysis import find_neighbor_papers

        service = MagicMock()
        service.search_by_id.return_value = [(self.excluded, 0.9), (self.outsider, 0.8)]

        result = find_neighbor_papers([self.included], exclude_ids={self.excluded}, embedding_service=service)

        self.assertEqual([r["id"] for r in result["results"]], [self.outsider])

    def test_mcp_collection_tools_drop_excluded_unless_asked(self):
        result = mcp_tools.get_collection(self.cid)
        self.assertEqual(result["count"], 3)
        decisions = {p["id"]: p["decision"] for p in result["papers"]}
        self.assertEqual(decisions, {self.unscreened: None, self.included: "include", self.maybe: "maybe"})

        everything = mcp_tools.get_collection(self.cid, include_excluded=True)
        self.assertEqual(everything["count"], 4)
        self.assertEqual({p["id"]: p["decision"] for p in everything["papers"]}[self.excluded], "exclude")

        listed = {c["id"]: c["paper_count"] for c in mcp_tools.list_collections()["collections"]}
        self.assertEqual(listed[self.cid], 3)

    # ── bundles ──

    def test_bundle_carries_decisions_fill_only(self):
        from app.services.collection_share import export_collection, import_collection

        bundle = export_collection(self.cid)
        self.assertEqual([p["decision"] for p in bundle["papers"]], [None, "include", "maybe", "exclude"])

        # Into a collection that already screened two of them: local decisions win,
        # blanks fill, and junk values from the untrusted bundle are dropped.
        target = Collection(name="Target")
        db.session.add(target)
        db.session.flush()
        db.session.add(PaperCollection(paper_id=self.included, collection_id=target.id, decision="exclude"))
        db.session.add(PaperCollection(paper_id=self.maybe, collection_id=target.id))
        db.session.commit()
        bundle["papers"][0]["decision"] = "bogus"
        import_collection(bundle, into=target)  # every paper is local: no fetch, no embedding

        got = {pc.paper_id: pc.decision for pc in PaperCollection.query.filter_by(collection_id=target.id)}
        self.assertEqual(
            got, {self.unscreened: None, self.included: "exclude", self.maybe: "maybe", self.excluded: "exclude"}
        )
