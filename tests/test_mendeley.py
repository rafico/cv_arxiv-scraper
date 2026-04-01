"""Tests for Mendeley API client (all mocked)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.mendeley import MendeleyClient


def _make_paper():
    """Create a mock Paper object for testing."""
    paper = Mock()
    paper.title = "Test Paper"
    paper.authors = "Alice Smith, Bob Jones"
    paper.arxiv_id = "2603.12345"
    paper.link = "https://arxiv.org/abs/2603.12345"
    paper.pdf_link = "https://arxiv.org/pdf/2603.12345"
    paper.abstract_text = "An abstract."
    paper.publication_dt = date(2026, 3, 13)
    return paper


class MendeleyClientTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.creds_path = self.tmpdir / "mendeley_credentials.json"
        self.token_path = self.tmpdir / ".mendeley_token"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write_creds(self):
        self.creds_path.write_text(
            json.dumps({"client_id": "test-id", "client_secret": "test-secret"}),
            encoding="utf-8",
        )

    def _write_token(self, token="test-access-token"):
        self.token_path.write_text(
            json.dumps({"access_token": token}),
            encoding="utf-8",
        )

    def _client(self):
        return MendeleyClient(
            credentials_path=self.creds_path,
            token_path=self.token_path,
        )

    @patch("app.services.mendeley.requests.post")
    def test_add_document_posts_to_api(self, mock_post):
        self._write_creds()
        self._write_token()
        mock_post.return_value = Mock(status_code=201)
        mock_post.return_value.raise_for_status = Mock()
        mock_post.return_value.json.return_value = {"id": "doc-123"}

        client = self._client()
        result = client.add_document(_make_paper())

        self.assertTrue(result["success"])
        self.assertEqual(result["document_id"], "doc-123")
        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        self.assertIn("/documents", call_url)

    @patch("app.services.mendeley.requests.get")
    def test_check_connection_returns_status(self, mock_get):
        self._write_creds()
        self._write_token()
        mock_get.return_value = Mock(status_code=200)

        client = self._client()
        result = client.check_connection()

        self.assertEqual(result["status"], "connected")

    def test_oauth_flow_start_returns_url(self):
        self._write_creds()
        client = self._client()
        result = client.start_oauth_flow(redirect_uri="http://localhost/callback")

        self.assertTrue(result["success"])
        self.assertIn("api.mendeley.com", result["auth_url"])
        self.assertIsNotNone(result["state"])

    @patch("app.services.mendeley.requests.post")
    def test_oauth_flow_finish_saves_token(self, mock_post):
        self._write_creds()
        mock_post.return_value = Mock(status_code=200)
        mock_post.return_value.raise_for_status = Mock()
        mock_post.return_value.json.return_value = {
            "access_token": "new-token",
            "refresh_token": "new-refresh",
        }

        client = self._client()
        result = client.finish_oauth_flow(
            authorization_response_url="http://localhost/callback?code=abc123&state=xyz",
            redirect_uri="http://localhost/callback",
        )

        self.assertTrue(result["success"])
        self.assertTrue(self.token_path.exists())
        saved = json.loads(self.token_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["access_token"], "new-token")

    @patch("app.services.mendeley.requests.post")
    def test_add_document_maps_paper_fields_correctly(self, mock_post):
        self._write_creds()
        self._write_token()
        mock_post.return_value = Mock(status_code=201)
        mock_post.return_value.raise_for_status = Mock()
        mock_post.return_value.json.return_value = {"id": "doc-456"}

        paper = _make_paper()
        client = self._client()
        client.add_document(paper)

        call_kwargs = mock_post.call_args
        doc = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        self.assertEqual(doc["title"], "Test Paper")
        self.assertIn("https://arxiv.org/abs/2603.12345", doc["websites"])
        self.assertEqual(doc["identifiers"]["arxiv"], "2603.12345")
        self.assertEqual(doc["year"], 2026)

    @patch("app.services.mendeley.requests.get")
    def test_check_connection_returns_error_on_expired_token(self, mock_get):
        self._write_creds()
        self._write_token()
        mock_get.return_value = Mock(status_code=401)

        client = self._client()
        result = client.check_connection()

        self.assertEqual(result["status"], "expired")

    def test_check_connection_no_credentials(self):
        client = self._client()
        result = client.check_connection()
        self.assertEqual(result["status"], "no_credentials")

    def test_check_connection_no_token(self):
        self._write_creds()
        client = self._client()
        result = client.check_connection()
        self.assertEqual(result["status"], "no_token")

    def test_start_oauth_no_credentials(self):
        client = self._client()
        result = client.start_oauth_flow("http://localhost/callback")
        self.assertFalse(result["success"])
