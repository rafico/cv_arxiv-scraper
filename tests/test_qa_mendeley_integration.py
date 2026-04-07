from __future__ import annotations

from unittest.mock import patch

from app.models import Paper, PaperFeedback, db
from tests.helpers import FlaskDBTestCase


class MendeleyIntegrationQaTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def _make_paper(self, *, title: str, arxiv_id: str, mendeley_doc_id: str | None = None) -> Paper:
        paper = Paper(
            arxiv_id=arxiv_id,
            title=title,
            authors="Alice Smith, Bob Jones",
            link=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_link=f"https://arxiv.org/pdf/{arxiv_id}",
            abstract_text="Paper abstract",
            topic_tags=["Vision"],
            match_type="Title",
            matched_terms=["Vision"],
            paper_score=1.0,
            publication_date="2026-04-07",
            scraped_date="2026-04-07",
            mendeley_doc_id=mendeley_doc_id,
        )
        db.session.add(paper)
        db.session.flush()
        db.session.add(PaperFeedback(paper_id=paper.id, action="save"))
        return paper

    @patch("app.services.mendeley.MendeleyClient.add_document")
    @patch("app.services.mendeley.MendeleyClient.check_connection")
    def test_bulk_sync_persists_doc_ids_and_skips_existing_links(self, mock_check_connection, mock_add_document):
        mock_check_connection.return_value = {"status": "connected", "message": "Mendeley is connected."}
        mock_add_document.side_effect = [
            {"success": True, "message": "Document added to Mendeley.", "document_id": "doc-101"},
            {"success": True, "message": "Document added to Mendeley.", "document_id": "doc-102"},
        ]

        new_one = self._make_paper(title="New One", arxiv_id="2604.0001")
        new_two = self._make_paper(title="New Two", arxiv_id="2604.0002")
        already_synced = self._make_paper(title="Old One", arxiv_id="2604.0003", mendeley_doc_id="doc-existing")
        db.session.commit()

        response = self.client.post(
            "/settings/mendeley-sync",
            data={"csrf_token": self._csrf_token()},
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_add_document.call_count, 2)
        synced_ids = [call.args[0].id for call in mock_add_document.call_args_list]
        self.assertEqual(synced_ids, [new_one.id, new_two.id])

        db.session.expire_all()
        self.assertEqual(db.session.get(Paper, new_one.id).mendeley_doc_id, "doc-101")
        self.assertEqual(db.session.get(Paper, new_two.id).mendeley_doc_id, "doc-102")
        self.assertEqual(db.session.get(Paper, already_synced.id).mendeley_doc_id, "doc-existing")
        self.assertIn(b"Synced 2 new papers to Mendeley", response.data)
        self.assertIn(b"1 already linked", response.data)

    @patch("app.services.mendeley.MendeleyClient.check_connection")
    def test_mendeley_status_endpoint_returns_current_status(self, mock_check_connection):
        mock_check_connection.return_value = {"status": "connected", "message": "Mendeley is connected."}

        response = self.client.get("/settings/mendeley-status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "connected")
        mock_check_connection.assert_called_once()

    @patch("app.services.mendeley.MendeleyClient.add_document")
    @patch("app.services.mendeley.MendeleyClient.check_connection")
    def test_single_paper_export_persists_document_id(self, mock_check_connection, mock_add_document):
        mock_check_connection.return_value = {"status": "connected", "message": "Mendeley is connected."}
        mock_add_document.return_value = {
            "success": True,
            "message": "Document added to Mendeley.",
            "document_id": "doc-303",
        }

        paper = self._make_paper(title="Single Paper", arxiv_id="2604.0303")
        db.session.commit()

        response = self.client.post(
            f"/api/papers/{paper.id}/mendeley",
            json={},
            headers={"X-CSRF-Token": self._csrf_token()},
        )

        self.assertEqual(response.status_code, 200)
        db.session.expire_all()
        self.assertEqual(db.session.get(Paper, paper.id).mendeley_doc_id, "doc-303")
