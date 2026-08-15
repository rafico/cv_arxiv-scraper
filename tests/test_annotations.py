from __future__ import annotations

from app.models import Paper, PaperAnnotation, db
from tests.helpers import FlaskDBTestCase

RECT = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.05}


def _paper() -> Paper:
    return Paper(
        arxiv_id="2601.00001",
        title="Annotated Paper",
        authors="Author A",
        link="https://arxiv.org/abs/2601.00001",
        pdf_link="https://arxiv.org/pdf/2601.00001.pdf",
        match_type="title",
        matched_terms=["Vision"],
        paper_score=1.0,
        publication_date="2026-01-01",
        scraped_date="2026-01-01",
    )


class AnnotationApiTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        self.paper = _paper()
        db.session.add(self.paper)
        db.session.commit()

    def _csrf(self) -> dict:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return {"X-CSRF-Token": session["settings_csrf_token"]}

    def _create(self, **overrides) -> dict:
        payload = {"kind": "highlight", "page": 1, "rects": [RECT], "color": "#ffd54a"}
        payload.update(overrides)
        response = self.client.post(
            f"/api/papers/{self.paper.id}/annotations", json=payload, headers=self._csrf()
        )
        return response

    def test_crud_roundtrip(self):
        created = self._create(kind="comment", rects=[{"x": 0.5, "y": 0.5, "w": 0, "h": 0}], note="check eq. 3")
        self.assertEqual(created.status_code, 201)
        annotation_id = created.get_json()["id"]

        listed = self.client.get(f"/api/papers/{self.paper.id}/annotations").get_json()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["note"], "check eq. 3")

        patched = self.client.patch(
            f"/api/annotations/{annotation_id}", json={"note": "resolved"}, headers=self._csrf()
        )
        self.assertEqual(patched.status_code, 200)
        self.assertEqual(patched.get_json()["note"], "resolved")

        deleted = self.client.delete(f"/api/annotations/{annotation_id}", headers=self._csrf())
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(PaperAnnotation.query.count(), 0)

    def test_validation_rejects_bad_input(self):
        self.assertEqual(self._create(kind="underline").status_code, 400)
        self.assertEqual(self._create(page=0).status_code, 400)
        self.assertEqual(self._create(page=True).status_code, 400)
        self.assertEqual(self._create(rects=[]).status_code, 400)
        self.assertEqual(self._create(rects=[{"x": 0.1, "y": 0.2, "w": 1.5, "h": 0.1}]).status_code, 400)
        self.assertEqual(self._create(rects=[{"x": 0.1, "y": 0.2}]).status_code, 400)
        self.assertEqual(self._create(note="x" * 20_001).status_code, 400)
        self.assertEqual(PaperAnnotation.query.count(), 0)

    def test_mutations_require_csrf(self):
        response = self.client.post(
            f"/api/papers/{self.paper.id}/annotations",
            json={"kind": "highlight", "page": 1, "rects": [RECT]},
        )
        self.assertEqual(response.status_code, 400)

    def test_unknown_paper_or_annotation_404s(self):
        self.assertEqual(self.client.get("/api/papers/9999/annotations").status_code, 404)
        self.assertEqual(
            self.client.patch("/api/annotations/9999", json={"note": "x"}, headers=self._csrf()).status_code, 404
        )

    def test_annotations_travel_in_collection_bundles(self):
        from app.models import Collection, PaperCollection
        from app.services.collection_share import export_collection, import_collection

        self._create(note="travels")
        collection = Collection(name="With Annotations")
        db.session.add(collection)
        db.session.flush()
        db.session.add(PaperCollection(paper_id=self.paper.id, collection_id=collection.id))
        db.session.commit()

        manifest = export_collection(collection.id)
        self.assertEqual(len(manifest["annotations"]), 1)
        self.assertEqual(manifest["annotations"][0]["paper"], 0)

        # Same-DB import: the linked paper already has this annotation — no duplicate.
        import_collection(manifest)
        self.assertEqual(PaperAnnotation.query.count(), 1)

        # Fresh-DB import: annotation is recreated on the new paper row.
        db.session.query(PaperAnnotation).delete()
        db.session.query(PaperCollection).delete()
        db.session.query(Paper).delete()
        db.session.query(Collection).delete()
        db.session.commit()
        import_collection(manifest)
        annotation = PaperAnnotation.query.one()
        self.assertEqual(annotation.note, "travels")
        self.assertEqual(annotation.rects, [RECT])
