from __future__ import annotations

from app.models import Collection, Paper, PaperCollection, PaperRelation, db
from app.services.collection_share import export_collection, import_collection
from tests.helpers import FlaskDBTestCase


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
        self.assertEqual(stats, {"created": 0, "linked": 2, "edges": 1})  # edge W1→W2 synced
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
