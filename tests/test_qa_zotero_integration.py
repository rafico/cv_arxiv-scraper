from __future__ import annotations

from unittest.mock import patch

from tests.helpers import FlaskDBTestCase


class ZoteroIntegrationQaTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    @patch("app.services.zotero.ZoteroClient.check_connection")
    @patch("app.services.zotero.ZoteroClient._save_credentials")
    def test_zotero_setup_saves_credentials_from_settings(self, mock_save, mock_check):
        mock_check.return_value = {"status": "connected", "message": "Zotero is connected."}

        response = self.client.post(
            "/settings/zotero-setup",
            data={
                "csrf_token": self._csrf_token(),
                "zotero_api_key": "api-key-123",
                "zotero_user_id": "user-456",
            },
        )

        self.assertEqual(response.status_code, 302)
        mock_save.assert_called_once_with("api-key-123", "user-456")
        mock_check.assert_called_once_with()

    @patch("app.services.zotero.ZoteroClient.check_connection")
    def test_zotero_test_endpoint_runs_status_check(self, mock_check):
        mock_check.return_value = {"status": "connected", "message": "Zotero is connected."}

        response = self.client.post(
            "/settings/zotero-test",
            data={"csrf_token": self._csrf_token()},
        )

        self.assertEqual(response.status_code, 302)
        mock_check.assert_called_once_with()

    @patch("app.services.zotero.ZoteroClient.list_collections")
    def test_zotero_collections_endpoint_returns_collection_list(self, mock_list):
        mock_list.return_value = [
            {"key": "COLL1", "name": "Vision"},
            {"key": "COLL2", "name": "Robotics"},
        ]

        response = self.client.get("/settings/zotero-collections")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), mock_list.return_value)
