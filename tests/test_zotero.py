"""Tests for Zotero API client (all mocked)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.zotero import ZoteroClient


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


class ZoteroClientTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.creds_path = self.tmpdir / ".zotero_credentials"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write_creds(self, api_key="test-key", user_id="12345"):
        self.creds_path.write_text(
            json.dumps({"api_key": api_key, "user_id": user_id}),
            encoding="utf-8",
        )

    def _client(self):
        return ZoteroClient(credentials_path=self.creds_path)

    @patch("app.services.zotero.requests.post")
    def test_add_item_posts_to_api(self, mock_post):
        self._write_creds()
        mock_post.return_value = Mock(status_code=200)
        mock_post.return_value.raise_for_status = Mock()

        client = self._client()
        result = client.add_item(_make_paper())

        self.assertTrue(result["success"])
        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        self.assertIn("/users/12345/items", call_url)

    @patch("app.services.zotero.requests.post")
    def test_add_item_maps_paper_fields_to_zotero_schema(self, mock_post):
        self._write_creds()
        mock_post.return_value = Mock(status_code=200)
        mock_post.return_value.raise_for_status = Mock()

        client = self._client()
        client.add_item(_make_paper())

        call_kwargs = mock_post.call_args
        items = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["itemType"], "journalArticle")
        self.assertEqual(item["title"], "Test Paper")
        self.assertEqual(item["url"], "https://arxiv.org/abs/2603.12345")
        self.assertIn("arXiv:2603.12345", item["extra"])
        self.assertEqual(len(item["creators"]), 2)
        self.assertEqual(item["creators"][0]["lastName"], "Smith")
        self.assertEqual(item["creators"][1]["lastName"], "Jones")

    @patch("app.services.zotero.requests.get")
    def test_check_connection_returns_status(self, mock_get):
        self._write_creds()
        mock_get.return_value = Mock(status_code=200)

        client = self._client()
        result = client.check_connection()

        self.assertEqual(result["status"], "connected")

    @patch("app.services.zotero.requests.get")
    def test_check_connection_returns_error_on_invalid_key(self, mock_get):
        self._write_creds()
        mock_get.return_value = Mock(status_code=403)

        client = self._client()
        result = client.check_connection()

        self.assertEqual(result["status"], "invalid")

    @patch("app.services.zotero.requests.get")
    def test_list_collections_returns_parsed_list(self, mock_get):
        self._write_creds()
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.raise_for_status = Mock()
        mock_get.return_value.json.return_value = [
            {"key": "ABC123", "data": {"name": "My Collection"}},
            {"key": "DEF456", "data": {"name": "Another"}},
        ]

        client = self._client()
        collections = client.list_collections()

        self.assertEqual(len(collections), 2)
        self.assertEqual(collections[0]["key"], "ABC123")
        self.assertEqual(collections[0]["name"], "My Collection")

    @patch("app.services.zotero.requests.post")
    def test_sync_saved_papers_batches_requests(self, mock_post):
        self._write_creds()
        mock_post.return_value = Mock(status_code=200)
        mock_post.return_value.raise_for_status = Mock()

        papers = [_make_paper() for _ in range(75)]
        client = self._client()
        result = client.sync_saved_papers(papers)

        self.assertTrue(result["success"])
        self.assertEqual(result["synced_count"], 75)
        # Should be 2 batches: 50 + 25
        self.assertEqual(mock_post.call_count, 2)

        first_batch = mock_post.call_args_list[0].kwargs.get("json") or mock_post.call_args_list[0][1].get("json")
        second_batch = mock_post.call_args_list[1].kwargs.get("json") or mock_post.call_args_list[1][1].get("json")
        self.assertEqual(len(first_batch), 50)
        self.assertEqual(len(second_batch), 25)

    @patch("app.services.zotero.requests.post")
    def test_add_item_with_collection_key(self, mock_post):
        self._write_creds()
        mock_post.return_value = Mock(status_code=200)
        mock_post.return_value.raise_for_status = Mock()

        client = self._client()
        client.add_item(_make_paper(), collection_key="COL123")

        items = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        self.assertEqual(items[0]["collections"], ["COL123"])

    def test_check_connection_no_credentials(self):
        client = self._client()
        result = client.check_connection()
        self.assertEqual(result["status"], "no_credentials")

    def test_save_credentials(self):
        client = self._client()
        client._save_credentials("my-key", "99999")
        self.assertTrue(self.creds_path.exists())
        data = json.loads(self.creds_path.read_text(encoding="utf-8"))
        self.assertEqual(data["api_key"], "my-key")
        self.assertEqual(data["user_id"], "99999")

    @patch("app.services.zotero.requests.post")
    def test_sync_empty_list(self, mock_post):
        self._write_creds()
        client = self._client()
        result = client.sync_saved_papers([])
        self.assertTrue(result["success"])
        self.assertEqual(result["synced_count"], 0)
        mock_post.assert_not_called()
