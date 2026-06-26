"""One-click backup/restore: consistent DB snapshot + FAISS index + config.yaml.

These functions take explicit paths (no Flask), so they are unit-testable without
an app context. The HTTP layer in ``app/routes/api/backup.py`` resolves the paths
from ``current_app`` and calls in here.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

# Archive member names (kept stable so older/newer backups stay readable).
_DB_MEMBER = "arxiv_papers.db"
_FAISS_PREFIX = "faiss_index/"
_CONFIG_MEMBER = "config.yaml"
_METADATA_MEMBER = "metadata.json"


def _snapshot_database(db_path: Path, dest_path: Path) -> bool:
    """Copy ``db_path`` into ``dest_path`` via SQLite's online backup API.

    Using the backup API (rather than a raw file copy) yields a consistent
    snapshot even while WAL writes are in flight. Returns ``True`` when a snapshot
    was written, ``False`` when the source DB does not exist yet.
    """
    if not db_path.exists():
        return False
    source = sqlite3.connect(str(db_path))
    try:
        dest = sqlite3.connect(str(dest_path))
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()
    return True


def create_backup(
    *,
    db_path: str | os.PathLike[str],
    faiss_dir: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    app_version: str = "0.1.0",
    created_at: str | None = None,
) -> bytes:
    """Build a ``.tar.gz`` snapshot of the DB, FAISS index dir, and config.

    Returns the archive as bytes. ``created_at`` defaults to the current UTC time
    in ISO-8601; pass an explicit value to make output deterministic in tests.
    """
    db_path = Path(db_path)
    faiss_dir = Path(faiss_dir)
    config_path = Path(config_path)
    created_at = created_at or datetime.now(timezone.utc).isoformat()

    contents: list[str] = []
    buffer = io.BytesIO()

    with tempfile.TemporaryDirectory() as tmp_dir:
        snapshot_path = Path(tmp_dir) / _DB_MEMBER
        has_db = _snapshot_database(db_path, snapshot_path)

        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            if has_db:
                tar.add(snapshot_path, arcname=_DB_MEMBER)
                contents.append(_DB_MEMBER)

            if faiss_dir.is_dir():
                for entry in sorted(faiss_dir.iterdir()):
                    if entry.is_file():
                        arcname = _FAISS_PREFIX + entry.name
                        tar.add(entry, arcname=arcname)
                        contents.append(arcname)

            if config_path.is_file():
                tar.add(config_path, arcname=_CONFIG_MEMBER)
                contents.append(_CONFIG_MEMBER)

            metadata = {
                "schema_version": SCHEMA_VERSION,
                "app_version": app_version,
                "created_at": created_at,
                "contents": contents,
            }
            meta_bytes = json.dumps(metadata, indent=2).encode("utf-8")
            info = tarfile.TarInfo(name=_METADATA_MEMBER)
            info.size = len(meta_bytes)
            tar.addfile(info, io.BytesIO(meta_bytes))

    return buffer.getvalue()


def _is_within(base: Path, target: Path) -> bool:
    """True when ``target`` resolves to a path inside ``base`` (or is ``base``)."""
    try:
        target.relative_to(base)
    except ValueError:
        return False
    return True


def _safe_extract(tar: tarfile.TarFile, dest_dir: Path) -> None:
    """Extract ``tar`` into ``dest_dir`` with path-traversal protection.

    Rejects any member that is absolute, contains a ``..`` component, or whose
    resolved destination escapes ``dest_dir``. ``tarfile.data_filter`` is only
    available on Python 3.12+, so the guard is implemented explicitly to keep the
    3.10 floor.
    """
    dest_root = dest_dir.resolve()
    for member in tar.getmembers():
        name = member.name
        member_path = Path(name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe path in archive: {name!r}")
        resolved = (dest_root / member_path).resolve()
        if not _is_within(dest_root, resolved):
            raise ValueError(f"Unsafe path in archive: {name!r}")
        # Reject symlinks/hardlinks/devices: only regular files and dirs belong here.
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Unsupported archive member type: {name!r}")
    # Members are validated above (no traversal, no links); safe to extract.
    # Pass the stdlib "data" filter on 3.12+ as defence-in-depth; the kwarg does
    # not exist on the 3.10 floor, where our explicit guard above is the safeguard.
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest_dir, filter="data")  # noqa: S202 - members validated by _safe_extract guard
    else:
        tar.extractall(dest_dir)  # noqa: S202 - members validated by _safe_extract guard


def _remove_wal_sidecars(db_path: Path) -> None:
    """Delete stale ``-wal`` / ``-shm`` sidecars next to a restored DB file."""
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(db_path.name + suffix)
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass


def _atomic_write(target: Path, data: bytes) -> None:
    """Write ``data`` to ``target`` atomically (tempfile + replace, in-place fallback).

    Mirrors ``app.services.preferences.save_config``: the rename fallback covers a
    single-file bind mount where ``os.replace`` over the mount point fails.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=target.suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        try:
            os.replace(tmp_path, target)
        except OSError:
            with open(target, "wb") as handle:
                handle.write(data)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _swap_faiss_dir(restored_dir: Path, faiss_dir: Path) -> None:
    """Replace ``faiss_dir`` with ``restored_dir`` via a tempfile-sibling swap."""
    faiss_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=str(faiss_dir.parent), prefix=".faiss_new_"))
    # mkdtemp made an empty dir; replace it with the restored contents.
    staging.rmdir()
    os.replace(restored_dir, staging)

    backup_old: Path | None = None
    if faiss_dir.exists():
        backup_old = Path(tempfile.mkdtemp(dir=str(faiss_dir.parent), prefix=".faiss_old_"))
        backup_old.rmdir()
        os.replace(faiss_dir, backup_old)
    try:
        os.replace(staging, faiss_dir)
    except OSError:
        # Roll back to the previous dir if the final swap fails.
        if backup_old is not None and backup_old.exists():
            os.replace(backup_old, faiss_dir)
        raise
    else:
        if backup_old is not None:
            _rmtree(backup_old)


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def restore_backup(
    archive_bytes: bytes,
    *,
    db_path: str | os.PathLike[str],
    faiss_dir: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
) -> dict:
    """Restore a backup archive over the live DB, FAISS index, and config.

    Raises ``ValueError`` on malformed or unsupported archives. On success the
    restored files are swapped into place atomically and a summary dict is
    returned.
    """
    db_path = Path(db_path)
    faiss_dir = Path(faiss_dir)
    config_path = Path(config_path)

    restored: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        extract_dir = Path(tmp_dir)
        try:
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
                _safe_extract(tar, extract_dir)
        except tarfile.TarError as exc:
            raise ValueError(f"Not a valid backup archive: {exc}") from exc

        metadata_path = extract_dir / _METADATA_MEMBER
        if not metadata_path.is_file():
            raise ValueError("Archive is missing metadata.json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"Malformed metadata.json: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError("Malformed metadata.json: expected an object")
        schema_version = metadata.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported backup schema_version: {schema_version!r}")

        # Database: atomic replace, then drop stale WAL sidecars from the old DB.
        restored_db = extract_dir / _DB_MEMBER
        if restored_db.is_file():
            db_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(restored_db, db_path)
            _remove_wal_sidecars(db_path)
            restored.append(_DB_MEMBER)

        # FAISS index dir: swap the whole directory atomically.
        restored_faiss = extract_dir / "faiss_index"
        if restored_faiss.is_dir():
            _swap_faiss_dir(restored_faiss, faiss_dir)
            restored.append("faiss_index/")

        # config.yaml: same atomic write approach as save_config().
        restored_config = extract_dir / _CONFIG_MEMBER
        if restored_config.is_file():
            _atomic_write(config_path, restored_config.read_bytes())
            restored.append(_CONFIG_MEMBER)

    return {
        "restored": restored,
        "schema_version": schema_version,
        "note": "Restart the app to fully reload the index/DB",
    }
