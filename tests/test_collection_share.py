from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.models import Collection, Paper, PaperCollection, PaperFeedback, PaperRelation, db
from app.services.collection_share import export_collection, import_collection
from tests.helpers import FlaskDBTestCase

# One resolvable entry from the arXiv API id_list query.
_ATOM_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1905.00001v2</id>
    <title>An Older Seed Paper</title>
    <summary>Seed abstract.</summary>
    <published>2019-05-01T00:00:00Z</published>
    <author><name>Dana Seed</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/1905.00001v2" rel="related" type="application/pdf"/>
    <category term="cs.CV" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""


def _paper(arxiv_id: str, **kwargs) -> Paper:
    defaults = dict(
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
    )
    defaults.update(kwargs)
    return Paper(**defaults)


class CollectionShareTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        backfill = patch("app.services.embed_backfill.backfill_embeddings", return_value=0)
        self.backfill = backfill.start()
        self.addCleanup(backfill.stop)
        embed = patch("app.services.embeddings.add_papers_to_index", return_value=0)
        self.embed = embed.start()
        self.addCleanup(embed.stop)
        service = patch("app.services.embeddings.get_embedding_service", return_value=MagicMock(index_dir="/idx"))
        service.start()
        self.addCleanup(service.stop)

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def _seed_collection(self) -> Collection:
        a = _paper("2601.00001", openalex_id="W1", referenced_works=["W2"], user_notes="my note", user_tags=["seg"])
        b = _paper("2601.00002", openalex_id="W2")
        outsider = _paper("2601.00003")
        collection = Collection(name="Bundle Source")
        db.session.add_all([a, b, outsider, collection])
        db.session.flush()
        db.session.add_all(
            [
                PaperCollection(paper_id=a.id, collection_id=collection.id),
                PaperCollection(paper_id=b.id, collection_id=collection.id),
            ]
        )
        db.session.commit()
        return collection

    def test_export_contains_members_only(self):
        collection = self._seed_collection()

        manifest = export_collection(collection.id)

        self.assertEqual(manifest["bundle_version"], 1)
        self.assertEqual(manifest["collection"]["name"], "Bundle Source")
        self.assertEqual([p["arxiv_id"] for p in manifest["papers"]], ["2601.00001", "2601.00002"])
        self.assertEqual(manifest["papers"][0]["user_notes"], "my note")
        self.assertEqual(manifest["papers"][0]["referenced_works"], ["W2"])

    def test_roundtrip_same_db_links_without_duplicating(self):
        collection = self._seed_collection()
        manifest = export_collection(collection.id)

        imported, stats = import_collection(manifest)

        self.assertEqual(imported.name, "Bundle Source (imported)")
        self.assertEqual(stats, {"created": 0, "linked": 2, "edges": 1, "embedded": True})  # edge W1→W2 synced
        self.assertEqual(Paper.query.count(), 3)  # no duplicate rows
        member_ids = {pc.paper_id for pc in PaperCollection.query.filter_by(collection_id=imported.id)}
        self.assertEqual(len(member_ids), 2)

    def test_import_into_empty_db_creates_papers_and_edges(self):
        collection = self._seed_collection()
        manifest = export_collection(collection.id)
        db.session.query(PaperCollection).delete()
        db.session.query(PaperRelation).delete()
        db.session.query(Paper).delete()
        db.session.query(Collection).delete()
        db.session.commit()

        imported, stats = import_collection(manifest)

        self.assertEqual(imported.name, "Bundle Source")
        self.assertEqual(stats["created"], 2)
        edge = PaperRelation.query.filter_by(relation_type="cites").one()
        citing = db.session.get(Paper, edge.paper_id)
        self.assertEqual(citing.arxiv_id, "2601.00001")
        self.assertEqual(citing.user_notes, "my note")
        self.assertEqual(citing.match_type, "import")

    def test_import_never_overwrites_local_notes(self):
        collection = self._seed_collection()
        manifest = export_collection(collection.id)
        local = Paper.query.filter_by(arxiv_id="2601.00001").one()
        local.user_notes = "precious local note"
        db.session.commit()

        import_collection(manifest)

        self.assertEqual(Paper.query.filter_by(arxiv_id="2601.00001").one().user_notes, "precious local note")

    def test_import_validation_rejects_garbage(self):
        for bad in (None, [], {}, {"bundle_version": 99}, {"bundle_version": 1, "collection": {"name": "x"}}):
            with self.assertRaises(ValueError):
                import_collection(bad)
        with self.assertRaises(ValueError):
            import_collection({"bundle_version": 1, "collection": {"name": "x"}, "papers": [{"title": "no link"}]})

    def test_api_roundtrip_and_csrf(self):
        collection = self._seed_collection()

        response = self.client.get(f"/api/collections/{collection.id}/export")
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment", response.headers["Content-Disposition"])
        manifest = response.get_json()

        no_csrf = self.client.post("/api/collections/import", json=manifest)
        self.assertEqual(no_csrf.status_code, 400)  # house CSRF failure code

        imported = self.client.post(
            "/api/collections/import", json=manifest, headers={"X-CSRF-Token": self._csrf_token()}
        )
        self.assertEqual(imported.status_code, 201)
        self.assertEqual(imported.get_json()["name"], "Bundle Source (imported)")

        malformed = self.client.post(
            "/api/collections/import",
            data="not json",
            content_type="application/json",
            headers={"X-CSRF-Token": self._csrf_token()},
        )
        self.assertEqual(malformed.status_code, 400)

        self.assertEqual(self.client.get("/api/collections/9999/export").status_code, 404)

    def test_import_into_existing_collection_embeds_new_papers(self):
        manifest = export_collection(self._seed_collection().id)
        db.session.query(PaperCollection).delete()
        db.session.query(Paper).delete()
        target = Collection(name="Review")
        db.session.add(target)
        db.session.commit()

        imported, stats = import_collection(manifest, into=target)

        self.assertEqual(imported.id, target.id)
        self.assertEqual(Collection.query.count(), 2)  # source + target, no "(imported)" copy
        self.assertEqual(stats["created"], 2)
        self.assertEqual(PaperCollection.query.filter_by(collection_id=target.id).count(), 2)
        self.embed.assert_called_once()

    def test_import_embeds_only_created_papers_through_the_locked_index_path(self):
        # Backfilling via the process singleton re-embedded the whole corpus and saved
        # its stale matrix over vectors a concurrent scrape or CLI had just written.
        manifest = export_collection(self._seed_collection().id)
        db.session.query(PaperCollection).delete()
        Paper.query.filter(Paper.arxiv_id != "2601.00001").delete()
        db.session.commit()

        import_collection(manifest)

        created = Paper.query.filter_by(arxiv_id="2601.00002").one()
        self.assertEqual(self.embed.call_args.args, ("/idx", [created.id], [f"{created.title} "]))
        self.backfill.assert_not_called()

    def test_bundle_route_leaves_large_imports_to_the_backfill(self):
        # Embedding a 5000-paper bundle would hold the single web worker for minutes.
        manifest = export_collection(self._seed_collection().id)
        db.session.query(PaperCollection).delete()
        db.session.query(Paper).delete()
        db.session.commit()

        with patch("app.routes.api.collections._MAX_IMPORT_IDS", 1):
            res = self.client.post(
                "/api/collections/import", json=manifest, headers={"X-CSRF-Token": self._csrf_token()}
            )

        self.assertEqual(res.status_code, 201)
        self.assertEqual((res.get_json()["created"], res.get_json()["embedded"]), (2, False))
        self.embed.assert_not_called()
        self.backfill.assert_not_called()

    def test_import_ids_reports_hidden_local_papers(self):
        # Hidden papers are filtered out of the collection view, .bib and MCP, so the
        # UI must say why a linked paper is missing instead of silently reloading.
        db.session.add(_paper("2601.00001", is_hidden=True))
        db.session.commit()

        res = self.client.post(
            "/api/collections/import-ids",
            json={"name": "Seeds", "text": "2601.00001"},
            headers={"X-CSRF-Token": self._csrf_token()},
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual((res.get_json()["linked"], res.get_json()["hidden"]), (1, ["2601.00001"]))

    def test_import_ids_seeds_collection_without_feedback(self):
        local = _paper("2601.00001")
        db.session.add(local)
        db.session.commit()
        text = (
            "@article{a, eprint={2601.00001}}\n"
            "https://arxiv.org/abs/1905.00001v2\n"
            "arXiv:1905.00002\n"
            "doi 10.1109/TPAMI.2019.2929257"
        )
        response = MagicMock(content=_ATOM_FEED)

        self.assertEqual(
            self.client.post("/api/collections/import-ids", json={"name": "Seeds", "text": text}).status_code, 400
        )
        with patch("app.services.ingest.arxiv_api_backend.request_with_backoff", return_value=response) as fetch:
            res = self.client.post(
                "/api/collections/import-ids",
                json={"name": "Seeds", "text": text},
                headers={"X-CSRF-Token": self._csrf_token()},
            )

        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual((body["created"], body["linked"], body["not_found"]), (1, 1, ["1905.00002"]))
        self.assertEqual(fetch.call_args.kwargs["params"]["id_list"], "1905.00001,1905.00002")  # local id not refetched
        collection = Collection.query.filter_by(name="Seeds").one()
        self.assertEqual(body["collection_id"], collection.id)
        members = Paper.query.join(PaperCollection).filter(PaperCollection.collection_id == collection.id).all()
        self.assertEqual({p.arxiv_id for p in members}, {"2601.00001", "1905.00001"})
        seed = Paper.query.filter_by(arxiv_id="1905.00001").one()
        self.assertEqual((seed.match_type, seed.authors), ("import", "Dana Seed"))
        self.assertEqual(PaperFeedback.query.count(), 0)  # never trains the ranker

    def test_import_ids_rejects_empty_and_oversized_input(self):
        token = self._csrf_token()
        for text, error in (
            ("no ids here", "No arXiv ids"),
            (" ".join(f"2401.{i:05d}" for i in range(101)), "max 100"),
        ):
            res = self.client.post(
                "/api/collections/import-ids", json={"name": "Seeds", "text": text}, headers={"X-CSRF-Token": token}
            )
            self.assertEqual(res.status_code, 400)
            self.assertIn(error, res.get_json()["error"])
        self.assertEqual(Collection.query.count(), 0)
