from __future__ import annotations

import errno
import io
import json
import os
import sqlite3
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Importing the route module attaches /api/backup/* to the shared api_bp blueprint
# before FlaskDBTestCase.setUp() registers the blueprint via create_app().
import app.routes.api.backup  # noqa: F401
from app.services.backup import SCHEMA_VERSION, create_backup, restore_backup
from tests.helpers import FlaskDBTestCase


def _make_db(path: Path, value: str) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO widgets (name) VALUES (?)", (value,))
        conn.commit()
    finally:
        conn.close()


def _read_db_names(path: Path) -> list[str]:
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute("SELECT name FROM widgets ORDER BY id").fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]


class BackupServiceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _build_source(self) -> tuple[Path, Path, Path]:
        src = self.root / "src"
        src.mkdir()
        db_path = src / "arxiv_papers.db"
        _make_db(db_path, "hello-row")

        faiss_dir = src / "faiss_index"
        faiss_dir.mkdir()
        (faiss_dir / "papers.index").write_bytes(b"\x00\x01\x02index-bytes")
        (faiss_dir / "id_map.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        config_path = src / "config.yaml"
        config_path.write_text("whitelists:\n  authors:\n    - Jane Doe\n", encoding="utf-8")
        return db_path, faiss_dir, config_path

    def test_round_trip_preserves_db_config_and_faiss(self):
        db_path, faiss_dir, config_path = self._build_source()
        archive = create_backup(
            db_path=db_path,
            faiss_dir=faiss_dir,
            config_path=config_path,
            created_at="2026-06-26T00:00:00+00:00",
        )
        self.assertEqual(archive[:2], b"\x1f\x8b")  # gzip magic bytes

        dest = self.root / "dest"
        dest.mkdir()
        new_db = dest / "arxiv_papers.db"
        new_faiss = dest / "faiss_index"
        new_config = dest / "config.yaml"

        summary = restore_backup(
            archive,
            db_path=new_db,
            faiss_dir=new_faiss,
            config_path=new_config,
        )

        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertIn("arxiv_papers.db", summary["restored"])
        self.assertIn("Restart the app", summary["note"])

        self.assertEqual(_read_db_names(new_db), ["hello-row"])
        self.assertEqual((new_faiss / "papers.index").read_bytes(), b"\x00\x01\x02index-bytes")
        self.assertEqual(json.loads((new_faiss / "id_map.json").read_text()), [1, 2, 3])
        self.assertIn("Jane Doe", new_config.read_text())

    def test_metadata_contents_and_created_at(self):
        db_path, faiss_dir, config_path = self._build_source()
        archive = create_backup(
            db_path=db_path,
            faiss_dir=faiss_dir,
            config_path=config_path,
            app_version="9.9.9",
            created_at="2026-06-26T12:34:56+00:00",
        )
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            meta = json.loads(tar.extractfile("metadata.json").read().decode("utf-8"))
        self.assertEqual(meta["schema_version"], SCHEMA_VERSION)
        self.assertEqual(meta["app_version"], "9.9.9")
        self.assertEqual(meta["created_at"], "2026-06-26T12:34:56+00:00")
        self.assertIn("arxiv_papers.db", meta["contents"])
        self.assertIn("config.yaml", meta["contents"])
        self.assertIn("faiss_index/papers.index", meta["contents"])

    def test_restore_survives_cross_device_tempdir(self):
        """Restore must work when the extraction tempdir is on another filesystem.

        Regression for the EXDEV crash: restore used to ``os.replace`` files
        straight out of the extraction tempdir into the instance dir, which fails
        with 'Invalid cross-device link' whenever /tmp is a separate mount (Docker,
        systemd PrivateTmp, tmpfs). We simulate a cross-fs boundary by rejecting any
        ``os.replace`` whose source and destination live in different directories —
        in this module those are exactly the extract->dest moves the bug performed;
        the fix only ever renames within the destination dir.
        """
        db_path, faiss_dir, config_path = self._build_source()
        archive = create_backup(
            db_path=db_path,
            faiss_dir=faiss_dir,
            config_path=config_path,
            created_at="2026-06-26T00:00:00+00:00",
        )
        dest = self.root / "dest"
        dest.mkdir()
        real_replace = os.replace

        def _exdev_replace(src, dst, *args, **kwargs):
            if os.path.dirname(os.path.realpath(src)) != os.path.dirname(os.path.realpath(dst)):
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real_replace(src, dst, *args, **kwargs)

        with patch("app.services.backup.os.replace", side_effect=_exdev_replace):
            summary = restore_backup(
                archive,
                db_path=dest / "arxiv_papers.db",
                faiss_dir=dest / "faiss_index",
                config_path=dest / "config.yaml",
            )

        self.assertIn("arxiv_papers.db", summary["restored"])
        self.assertEqual(_read_db_names(dest / "arxiv_papers.db"), ["hello-row"])
        self.assertEqual((dest / "faiss_index" / "papers.index").read_bytes(), b"\x00\x01\x02index-bytes")
        self.assertIn("Jane Doe", (dest / "config.yaml").read_text())

    def test_restore_rolls_back_when_a_later_step_fails(self):
        """A failure after the DB swap must leave the prior DB and FAISS intact.

        Restore is all-or-nothing: if a later component fails, the DB already
        swapped in is rolled back so the previous database is never lost.
        """
        db_path, faiss_dir, config_path = self._build_source()
        archive = create_backup(
            db_path=db_path,
            faiss_dir=faiss_dir,
            config_path=config_path,
            created_at="2026-06-26T00:00:00+00:00",
        )
        dest = self.root / "dest"
        dest.mkdir()
        new_db = dest / "arxiv_papers.db"
        new_faiss = dest / "faiss_index"
        new_config = dest / "config.yaml"
        # Pre-existing install that must survive a failed restore.
        _make_db(new_db, "OLD-ROW")
        new_faiss.mkdir()
        (new_faiss / "papers.index").write_bytes(b"OLD-INDEX")
        new_config.write_text("old: true\n", encoding="utf-8")

        # config.yaml is committed last; make that step fail.
        with patch("app.services.backup._atomic_write", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                restore_backup(archive, db_path=new_db, faiss_dir=new_faiss, config_path=new_config)

        # The old DB and FAISS index must be intact (rolled back), not the new ones.
        self.assertEqual(_read_db_names(new_db), ["OLD-ROW"])
        self.assertEqual((new_faiss / "papers.index").read_bytes(), b"OLD-INDEX")

    def test_rejects_oversized_archive(self):
        """A decompression bomb (huge declared size) is rejected before extraction."""
        db_path, faiss_dir, config_path = self._build_source()
        archive = create_backup(
            db_path=db_path,
            faiss_dir=faiss_dir,
            config_path=config_path,
            created_at="2026-06-26T00:00:00+00:00",
        )
        dest = self.root / "dest"
        dest.mkdir()
        # Shrink the budget so the (legitimate) archive trips the size guard,
        # standing in for a bomb whose declared member sizes exceed the cap.
        with patch("app.services.backup._MAX_EXTRACT_BYTES", 8):
            with self.assertRaises(ValueError) as ctx:
                restore_backup(
                    archive,
                    db_path=dest / "arxiv_papers.db",
                    faiss_dir=dest / "faiss_index",
                    config_path=dest / "config.yaml",
                )
        self.assertIn("too large", str(ctx.exception))
        # Nothing was written to the destination.
        self.assertFalse((dest / "arxiv_papers.db").exists())

    def test_restore_removes_stale_wal_sidecars(self):
        db_path, faiss_dir, config_path = self._build_source()
        archive = create_backup(
            db_path=db_path,
            faiss_dir=faiss_dir,
            config_path=config_path,
            created_at="2026-06-26T00:00:00+00:00",
        )
        dest = self.root / "dest"
        dest.mkdir()
        new_db = dest / "arxiv_papers.db"
        # Pre-existing stale sidecars from an older DB.
        new_db.with_name("arxiv_papers.db-wal").write_bytes(b"stale-wal")
        new_db.with_name("arxiv_papers.db-shm").write_bytes(b"stale-shm")

        restore_backup(
            archive,
            db_path=new_db,
            faiss_dir=dest / "faiss_index",
            config_path=dest / "config.yaml",
        )
        self.assertFalse(new_db.with_name("arxiv_papers.db-wal").exists())
        self.assertFalse(new_db.with_name("arxiv_papers.db-shm").exists())

    def _malicious_archive(self) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            payload = b"pwned"
            info = tarfile.TarInfo(name="../escape.txt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            meta = json.dumps({"schema_version": SCHEMA_VERSION}).encode("utf-8")
            minfo = tarfile.TarInfo(name="metadata.json")
            minfo.size = len(meta)
            tar.addfile(minfo, io.BytesIO(meta))
        return buffer.getvalue()

    def test_rejects_path_traversal_member(self):
        archive = self._malicious_archive()
        dest = self.root / "dest"
        dest.mkdir()
        with self.assertRaises(ValueError):
            restore_backup(
                archive,
                db_path=dest / "arxiv_papers.db",
                faiss_dir=dest / "faiss_index",
                config_path=dest / "config.yaml",
            )
        # Nothing should have escaped the destination dir.
        self.assertFalse((self.root / "escape.txt").exists())

    def test_rejects_missing_metadata(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            payload = b"x"
            info = tarfile.TarInfo(name="arxiv_papers.db")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        dest = self.root / "dest"
        dest.mkdir()
        with self.assertRaises(ValueError):
            restore_backup(
                buffer.getvalue(),
                db_path=dest / "arxiv_papers.db",
                faiss_dir=dest / "faiss_index",
                config_path=dest / "config.yaml",
            )

    def test_rejects_unsupported_schema_version(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            meta = json.dumps({"schema_version": 999}).encode("utf-8")
            info = tarfile.TarInfo(name="metadata.json")
            info.size = len(meta)
            tar.addfile(info, io.BytesIO(meta))
        dest = self.root / "dest"
        dest.mkdir()
        with self.assertRaises(ValueError):
            restore_backup(
                buffer.getvalue(),
                db_path=dest / "arxiv_papers.db",
                faiss_dir=dest / "faiss_index",
                config_path=dest / "config.yaml",
            )

    def test_rejects_non_gzip_archive(self):
        dest = self.root / "dest"
        dest.mkdir()
        with self.assertRaises(ValueError):
            restore_backup(
                b"not a tarball at all",
                db_path=dest / "arxiv_papers.db",
                faiss_dir=dest / "faiss_index",
                config_path=dest / "config.yaml",
            )


class BackupEndpointTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()

    def _csrf_token(self) -> str:
        self.client.get("/")
        with self.client.session_transaction() as session:
            return session["settings_csrf_token"]

    def test_export_returns_gzip_attachment(self):
        response = self.client.get("/api/backup/export")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data[:2], b"\x1f\x8b")  # gzip magic bytes
        disposition = response.headers.get("Content-Disposition", "")
        self.assertIn("attachment", disposition)
        self.assertIn("cv-arxiv-backup-", disposition)

    def test_import_requires_csrf(self):
        archive = io.BytesIO(b"ignored")
        response = self.client.post(
            "/api/backup/import",
            data={"backup": (archive, "backup.tar.gz")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)

    def test_import_missing_file_returns_400(self):
        token = self._csrf_token()
        response = self.client.post(
            "/api/backup/import",
            data={},
            headers={"X-CSRF-Token": token},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_export_then_import_round_trip(self):
        token = self._csrf_token()

        # Seed a known row so we can prove real DB data survives the HTTP round-trip
        # (not just that the response metadata looks right).
        from app.models import Paper, db

        db.session.add(
            Paper(
                arxiv_id="2606.99999",
                title="Round Trip",
                authors="A. Author",
                link="https://arxiv.org/abs/2606.99999",
                pdf_link="https://arxiv.org/pdf/2606.99999",
                match_type="Title",
                paper_score=1.0,
                publication_date="2026-01-01",
                scraped_date="2026-01-01",
            )
        )
        db.session.commit()

        export = self.client.get("/api/backup/export")
        self.assertEqual(export.status_code, 200)

        response = self.client.post(
            "/api/backup/import",
            data={"backup": (io.BytesIO(export.data), "backup.tar.gz")},
            headers={"X-CSRF-Token": token},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertIn("Restart the app", payload["note"])

        # The restored DB on disk must actually contain the seeded row.
        db_file = db.engine.url.database
        conn = sqlite3.connect(db_file)
        try:
            rows = conn.execute("SELECT arxiv_id FROM papers WHERE arxiv_id = ?", ("2606.99999",)).fetchall()
        finally:
            conn.close()
        self.assertEqual(rows, [("2606.99999",)])

    def test_import_os_error_returns_clean_500(self):
        """An OSError during restore is surfaced as JSON, not an unhandled 500."""
        token = self._csrf_token()
        export = self.client.get("/api/backup/export")
        with patch("app.routes.api.backup.restore_backup", side_effect=OSError("disk full")):
            response = self.client.post(
                "/api/backup/import",
                data={"backup": (io.BytesIO(export.data), "backup.tar.gz")},
                headers={"X-CSRF-Token": token},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main()
