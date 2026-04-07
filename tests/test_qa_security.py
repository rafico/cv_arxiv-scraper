"""QA tests for CSRF protection and credential safety.

Covers: CVARX-70 (Security)
- POST/PUT/DELETE without CSRF token returns 400
- Valid CSRF token allows mutation
- Credential files are gitignored
"""

from __future__ import annotations

import subprocess
from datetime import date, datetime, timezone

from app.models import Collection, Paper, db
from tests.helpers import FlaskDBTestCase


def _make_paper(**overrides) -> Paper:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    defaults = dict(
        arxiv_id="2607.9000",
        title="CSRF Test Paper",
        authors="Author A",
        link="https://arxiv.org/abs/2607.9000",
        pdf_link="https://arxiv.org/pdf/2607.9000",
        abstract_text="abstract",
        summary_text="summary",
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=10.0,
        feedback_score=0,
        is_hidden=False,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=now,
    )
    defaults.update(overrides)
    return Paper(**defaults)


class CsrfProtectionTests(FlaskDBTestCase):
    """Test that mutation endpoints reject requests without CSRF token."""

    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        paper = _make_paper()
        db.session.add(paper)
        db.session.commit()
        self.paper_id = paper.id

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_feedback_without_csrf_rejected(self):
        response = self.client.post(
            f"/api/papers/{self.paper_id}/feedback",
            json={"action": "save"},
        )
        self.assertEqual(response.status_code, 400)

    def test_feedback_with_csrf_accepted(self):
        token = self._csrf_token()
        response = self.client.post(
            f"/api/papers/{self.paper_id}/feedback",
            json={"action": "save"},
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 200)

    def test_collection_create_without_csrf_rejected(self):
        response = self.client.post(
            "/api/collections",
            json={"name": "No CSRF"},
        )
        self.assertEqual(response.status_code, 400)

    def test_collection_delete_without_csrf_rejected(self):
        c = Collection(name="Test")
        db.session.add(c)
        db.session.commit()
        response = self.client.delete(f"/api/collections/{c.id}")
        self.assertEqual(response.status_code, 400)

    def test_notes_without_csrf_rejected(self):
        response = self.client.put(
            f"/api/papers/{self.paper_id}/notes",
            json={"notes": "test"},
        )
        self.assertEqual(response.status_code, 400)

    def test_tags_without_csrf_rejected(self):
        response = self.client.post(
            f"/api/papers/{self.paper_id}/tags",
            json={"tag": "test"},
        )
        self.assertEqual(response.status_code, 400)

    def test_reading_status_without_csrf_rejected(self):
        response = self.client.post(
            f"/api/papers/{self.paper_id}/reading-status",
            json={"status": "to_read"},
        )
        self.assertEqual(response.status_code, 400)

    def test_wrong_csrf_token_rejected(self):
        self._csrf_token()  # Initialize session
        response = self.client.post(
            f"/api/papers/{self.paper_id}/feedback",
            json={"action": "save"},
            headers={"X-CSRF-Token": "wrong-token-value"},
        )
        self.assertEqual(response.status_code, 400)

    def test_bulk_feedback_without_csrf_rejected(self):
        response = self.client.post(
            "/api/papers/bulk-feedback",
            json={"paper_ids": [self.paper_id], "action": "save"},
        )
        self.assertEqual(response.status_code, 400)

    def test_feed_source_create_without_csrf_rejected(self):
        response = self.client.post(
            "/api/feed-sources",
            json={"name": "test", "url": "https://example.com/rss"},
        )
        self.assertEqual(response.status_code, 400)


class CredentialGitignoreTests(FlaskDBTestCase):
    """Verify that credential files are gitignored."""

    def test_credentials_json_gitignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "credentials.json"],
            capture_output=True,
            text=True,
            cwd=self.app.root_path + "/..",
        )
        self.assertEqual(result.returncode, 0)

    def test_token_json_gitignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "token.json"],
            capture_output=True,
            text=True,
            cwd=self.app.root_path + "/..",
        )
        self.assertEqual(result.returncode, 0)

    def test_mendeley_token_gitignored(self):
        result = subprocess.run(
            ["git", "check-ignore", ".mendeley_token"],
            capture_output=True,
            text=True,
            cwd=self.app.root_path + "/..",
        )
        self.assertEqual(result.returncode, 0)

    def test_zotero_credentials_gitignored(self):
        result = subprocess.run(
            ["git", "check-ignore", ".zotero_credentials"],
            capture_output=True,
            text=True,
            cwd=self.app.root_path + "/..",
        )
        self.assertEqual(result.returncode, 0)

    def test_llm_api_key_gitignored(self):
        result = subprocess.run(
            ["git", "check-ignore", ".llm_api_key"],
            capture_output=True,
            text=True,
            cwd=self.app.root_path + "/..",
        )
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
