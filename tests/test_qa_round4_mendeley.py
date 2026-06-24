"""QA round 4 regression tests for the Mendeley client (G3 / S2).

A stored ``.mendeley_token`` that is valid JSON but lacks ``access_token`` must
not crash ``check_connection()`` with an uncaught ``KeyError`` (which would 500
the /settings and /settings/mendeley-status routes). It should degrade
gracefully to an ``invalid`` status so the user can re-authorize. The same must
hold on the post-401 refresh path when the token endpoint returns a 2xx body
without ``access_token``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.mendeley import MendeleyClient


class MendeleyMissingAccessTokenTests(unittest.TestCase):
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

    def _client(self):
        return MendeleyClient(
            credentials_path=self.creds_path,
            token_path=self.token_path,
        )

    def test_g3_check_connection_token_missing_access_token_returns_invalid(self):
        # Valid JSON, but no access_token key — must not raise KeyError.
        self._write_creds()
        self.token_path.write_text(
            json.dumps({"refresh_token": "refresh-only", "token_type": "bearer"}),
            encoding="utf-8",
        )

        result = self._client().check_connection()

        self.assertEqual(result["status"], "invalid")
        self.assertIn("Re-authorize", result["message"])

    @patch("app.services.mendeley.requests.post")
    @patch("app.services.mendeley.requests.get")
    def test_g3_check_connection_refresh_body_missing_access_token_is_graceful(self, mock_get, mock_post):
        # access_token present initially so the GET happens; the GET 401s and the
        # refresh endpoint returns 2xx but WITHOUT access_token. Dereferencing
        # refreshed['access_token'] would raise KeyError without the fix.
        self._write_creds()
        self.token_path.write_text(
            json.dumps({"access_token": "stale", "refresh_token": "refresh-123"}),
            encoding="utf-8",
        )
        mock_get.return_value = Mock(status_code=401)
        mock_post.return_value = Mock(status_code=200)
        mock_post.return_value.raise_for_status = Mock()
        mock_post.return_value.json.return_value = {"refresh_token": "refresh-123"}

        result = self._client().check_connection()

        self.assertIn(result["status"], {"expired", "invalid"})
        self.assertNotEqual(result["status"], "connected")


if __name__ == "__main__":
    unittest.main()
