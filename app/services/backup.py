"""One-click backup/restore: consistent DB snapshot + FAISS index + config.yaml.

These functions take explicit paths (no Flask), so they are unit-testable without
an app context. The HTTP layer in ``app/routes/api/backup.py`` resolves the paths
from ``current_app`` and calls in here.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

# Upper bound on the total *decompressed* size of an imported archive. Caps a
# decompression bomb (a tiny gzip expanding to gigabytes) that would otherwise
# fill the disk during extraction — MAX_CONTENT_LENGTH only limits the compressed
# upload. The effective budget is the smaller of this ceiling and
# ``compressed_size * _MAX_COMPRESSION_RATIO`` (with a small floor), so legitimate
# backups always fit while pathological ratios are rejected before any bytes land.
_MAX_EXTRACT_BYTES = 1024 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 100
_MIN_EXTRACT_BUDGET = 16 * 1024 * 1024

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

        # Snapshot the FAISS index into the temp dir up front so the archive holds a
        # single point-in-time copy of the index files. Streaming them straight from
        # the live dir during the (longer) tar write could otherwise capture
        # papers.index and id_map.json from different generations, or a half-written
        # file, if a scrape rewrites the index concurrently — yielding a backup whose
        # index disagrees with its DB snapshot.
        faiss_snapshot: Path | None = None
        if faiss_dir.is_dir():
            faiss_snapshot = Path(tmp_dir) / "faiss_index"
            shutil.copytree(faiss_dir, faiss_snapshot)

        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            if has_db:
                tar.add(snapshot_path, arcname=_DB_MEMBER)
                contents.append(_DB_MEMBER)

            if faiss_snapshot is not None:
                for entry in sorted(faiss_snapshot.iterdir()):
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


def _safe_extract(tar: tarfile.TarFile, dest_dir: Path, *, max_total_bytes: int) -> None:
    """Extract ``tar`` into ``dest_dir`` with path-traversal and size protection.

    Rejects any member that is absolute, contains a ``..`` component, or whose
    resolved destination escapes ``dest_dir``; rejects symlink/device members; and
    rejects the archive outright when the declared total size of its file members
    exceeds ``max_total_bytes`` (decompression-bomb guard — checked from the tar
    headers before a single byte is written). ``tarfile.data_filter`` is only
    available on Python 3.12+, so the guards are implemented explicitly to keep the
    3.10 floor.
    """
    dest_root = dest_dir.resolve()
    total_bytes = 0
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
        if member.isfile():
            total_bytes += member.size
            if total_bytes > max_total_bytes:
                raise ValueError(
                    f"Backup archive too large to restore safely "
                    f"({total_bytes} bytes exceeds the {max_total_bytes}-byte limit)"
                )
    # Members are validated above (no traversal, no links, within size budget); extract.
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
    single-file bind mount where ``os.replace`` over the mount point fails. The
    tempfile lives next to ``target``, so the replace never crosses a filesystem.
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


def _unlink(path: Path) -> None:
    """Best-effort removal of a single file (staged / old-backup cleanup)."""
    try:
        path.unlink()
    except OSError:
        pass


def _rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _stage_file(src: Path, target: Path) -> Path:
    """Copy ``src`` onto ``target``'s filesystem; return the staged sibling path.

    The copy is the only cross-device-prone step, so it is done up front (before
    any destructive swap) into ``target.parent`` — guaranteeing the later commit is
    a same-filesystem ``os.replace`` that cannot fail with EXDEV.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, staged = tempfile.mkstemp(dir=str(target.parent), prefix="." + target.name + ".new_")
    os.close(fd)
    staged_path = Path(staged)
    shutil.copy2(src, staged_path)
    return staged_path


def _stage_dir(src: Path, target: Path) -> Path:
    """Copy directory ``src`` onto ``target``'s filesystem; return the staged path."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=str(target.parent), prefix="." + target.name + ".new_"))
    staging.rmdir()  # copytree requires a non-existent destination
    shutil.copytree(src, staging)
    return staging


def _commit_swap(staged: Path, target: Path, *, is_dir: bool, rollbacks: list, cleanups: list) -> None:
    """Swap a pre-staged sibling into ``target`` via same-filesystem renames.

    ``staged`` and ``target`` share a filesystem, so every rename here is a cheap,
    near-atomic ``os.replace``. The prior target (if any) is moved aside first and
    restored immediately if the final rename fails. On success a *rollback* closure
    (to undo this swap should a *later* component fail) and a *cleanup* closure (to
    drop the old backup) are recorded, giving the whole restore all-or-nothing
    semantics so the live DB is never left half-replaced.
    """
    old_backup: Path | None = None
    try:
        if target.exists():
            prefix = "." + target.name + ".old_"
            if is_dir:
                old_backup = Path(tempfile.mkdtemp(dir=str(target.parent), prefix=prefix))
                old_backup.rmdir()
            else:
                fd, ob = tempfile.mkstemp(dir=str(target.parent), prefix=prefix)
                os.close(fd)
                old_backup = Path(ob)
                old_backup.unlink()
            os.replace(target, old_backup)
        os.replace(staged, target)
    except OSError:
        # Restore this component's original state, then surface the error.
        if old_backup is not None and old_backup.exists():
            os.replace(old_backup, target)
        raise

    if old_backup is not None:
        backup = old_backup  # bind for the closures below

        def _rollback() -> None:
            # A non-empty dir cannot be a rename destination, so drop the new
            # target before restoring the old one.
            if target.exists():
                _rmtree(target) if is_dir else _unlink(target)
            os.replace(backup, target)

        rollbacks.append(_rollback)
        cleanups.append(lambda: _rmtree(backup) if is_dir else _unlink(backup))
    else:
        rollbacks.append(lambda: _rmtree(target) if is_dir else _unlink(target))


def _commit_removal(target: Path, *, is_dir: bool, rollbacks: list, cleanups: list) -> None:
    """Move ``target`` aside so the restore *omits* it, recoverably.

    Used when the archive restores a DB but carries no FAISS index: the live index
    belongs to the previous corpus and its row->paper_id map would reference paper ids
    the restored DB no longer has. We move it aside (same-fs rename, so cheap and
    near-atomic) rather than deleting outright, and register a rollback so a *later*
    component failure restores it — same all-or-nothing contract as ``_commit_swap``.
    """
    if not target.exists():
        return
    prefix = "." + target.name + ".old_"
    if is_dir:
        old_backup = Path(tempfile.mkdtemp(dir=str(target.parent), prefix=prefix))
        old_backup.rmdir()
    else:
        fd, ob = tempfile.mkstemp(dir=str(target.parent), prefix=prefix)
        os.close(fd)
        old_backup = Path(ob)
        old_backup.unlink()
    os.replace(target, old_backup)

    def _rollback() -> None:
        if target.exists():
            _rmtree(target) if is_dir else _unlink(target)
        os.replace(old_backup, target)

    rollbacks.append(_rollback)
    cleanups.append(lambda: _rmtree(old_backup) if is_dir else _unlink(old_backup))


def restore_backup(
    archive_bytes: bytes,
    *,
    db_path: str | os.PathLike[str],
    faiss_dir: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
) -> dict:
    """Restore a backup archive over the live DB, FAISS index, and config.

    Raises ``ValueError`` on malformed, unsupported, or oversized archives. The
    restore is staged then committed as a unit: every component is first copied
    onto its target filesystem (so the commit renames never cross a device
    boundary), then swapped in. If any swap fails, the components already swapped
    are rolled back, so the live DB is never left half-replaced.
    """
    db_path = Path(db_path)
    faiss_dir = Path(faiss_dir)
    config_path = Path(config_path)

    restored: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        extract_dir = Path(tmp_dir)
        max_total = min(_MAX_EXTRACT_BYTES, max(_MIN_EXTRACT_BUDGET, len(archive_bytes) * _MAX_COMPRESSION_RATIO))
        try:
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
                _safe_extract(tar, extract_dir, max_total_bytes=max_total)
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

        restored_db = extract_dir / _DB_MEMBER
        restored_faiss = extract_dir / "faiss_index"
        restored_config = extract_dir / _CONFIG_MEMBER

        # Phase 1 — stage each present component onto its target filesystem. These
        # copies are the only cross-device-prone work and run before anything
        # destructive, so a failure here leaves the live data untouched.
        staged_db = _stage_file(restored_db, db_path) if restored_db.is_file() else None
        staged_faiss = _stage_dir(restored_faiss, faiss_dir) if restored_faiss.is_dir() else None
        config_bytes = restored_config.read_bytes() if restored_config.is_file() else None

        # Phase 2 — commit the staged components, rolling back everything already
        # swapped if any step fails.
        rollbacks: list = []
        cleanups: list = []
        try:
            if staged_db is not None:
                _commit_swap(staged_db, db_path, is_dir=False, rollbacks=rollbacks, cleanups=cleanups)
                _remove_wal_sidecars(db_path)
                restored.append(_DB_MEMBER)
            if staged_faiss is not None:
                _commit_swap(staged_faiss, faiss_dir, is_dir=True, rollbacks=rollbacks, cleanups=cleanups)
                restored.append("faiss_index/")
            elif staged_db is not None:
                # DB restored but the archive carried no FAISS index (e.g. a backup taken
                # before the first scrape). Any live index belongs to the previous corpus
                # and would reference paper ids the restored DB no longer has, so drop it
                # atomically — the index can never disagree with the restored DB.
                _commit_removal(faiss_dir, is_dir=True, rollbacks=rollbacks, cleanups=cleanups)
                restored.append("faiss_index/ (cleared stale)")
            if config_bytes is not None:
                # config.yaml is committed last via the same atomic write as
                # save_config(); nothing follows it, so it needs no undo backup.
                _atomic_write(config_path, config_bytes)
                restored.append(_CONFIG_MEMBER)
        except BaseException:
            for undo in reversed(rollbacks):
                try:
                    undo()
                except OSError:
                    pass
            # Drop any staged artifacts that were never committed.
            if staged_db is not None:
                _unlink(staged_db)
            if staged_faiss is not None:
                _rmtree(staged_faiss)
            raise
        else:
            for done in cleanups:
                try:
                    done()
                except OSError:
                    pass

    return {
        "restored": restored,
        "schema_version": schema_version,
        "note": "Restart the app to fully reload the index/DB",
    }
