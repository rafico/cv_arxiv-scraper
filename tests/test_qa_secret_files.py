"""QA round 2 — secret files are created 0600 with no world-readable window.

The prior round fixed the write-then-chmod TOCTOU in llm_client only; the OAuth and
reference-manager secret writers still used ``write_text`` + ``chmod``, leaving the
token briefly world-readable. They now share ``write_secret_file``, which creates the
file with mode 0600 from the start. Each test neutralizes the trailing ``chmod`` (and
opens the umask) so the assertion proves the *creation* mode, not the tightening.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.services.secret_files import write_secret_file
from tests.helpers import FlaskDBTestCase


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class WriteSecretFileTests(FlaskDBTestCase):
    def test_writes_content_and_0600(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".secret"
            write_secret_file(path, "hunter2")
            self.assertEqual(path.read_text(encoding="utf-8"), "hunter2")
            self.assertEqual(_mode(path), 0o600)

    def test_created_0600_without_relying_on_chmod(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".secret"
            old_umask = os.umask(0)
            try:
                with patch("app.services.secret_files.os.chmod"):
                    write_secret_file(path, "hunter2")
                self.assertEqual(_mode(path), 0o600)
            finally:
                os.umask(old_umask)


class WriterTocTouTests(FlaskDBTestCase):
    """Each credential writer creates 0600 even with the trailing chmod removed."""

    def test_mendeley_token_created_0600_without_chmod(self):
        from app.services.mendeley import MendeleyClient

        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / ".mendeley_token"
            client = MendeleyClient(token_path=token_path)
            old_umask = os.umask(0)
            try:
                with patch("app.services.secret_files.os.chmod"):
                    client._save_token({"access_token": "a", "refresh_token": "r"})
                self.assertEqual(_mode(token_path), 0o600)
            finally:
                os.umask(old_umask)

    def test_zotero_credentials_created_0600_without_chmod(self):
        from app.services.zotero import ZoteroClient

        with tempfile.TemporaryDirectory() as tmpdir:
            creds_path = Path(tmpdir) / ".zotero_credentials"
            client = ZoteroClient(credentials_path=creds_path)
            old_umask = os.umask(0)
            try:
                with patch("app.services.secret_files.os.chmod"):
                    client._save_credentials("my-key", "99999")
                self.assertEqual(_mode(creds_path), 0o600)
            finally:
                os.umask(old_umask)
