"""One-click backup export/import endpoints."""

from datetime import datetime, timezone

from flask import Response, current_app, jsonify, request

from app.csrf import validate_csrf_token
from app.models import db
from app.routes.api import api_bp
from app.services.backup import create_backup, restore_backup

# The global MAX_CONTENT_LENGTH (app/__init__.py, 2 MiB) is sized for small
# credential/config uploads and would 413 any real backup — the SQLite DB alone is
# tens of MB. Raise the limit for this one route so restore_backup's own size /
# decompression-ratio guard is the real limiter. Bounded (not disabled) because the
# view reads the whole body into memory. Aligned with backup._MAX_EXTRACT_BYTES.
_MAX_BACKUP_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GiB


def _resolve_paths() -> tuple[str, str, str]:
    """Resolve (db_path, faiss_dir, config_path) from the active app."""
    db_path = db.engine.url.database
    faiss_dir = current_app.config["FAISS_INDEX_DIR"]
    config_path = current_app.config["CONFIG_PATH"]
    return db_path, faiss_dir, config_path


@api_bp.route("/backup/export", methods=["GET"])
def backup_export():
    db_path, faiss_dir, config_path = _resolve_paths()
    created_at = datetime.now(timezone.utc)
    archive = create_backup(
        db_path=db_path,
        faiss_dir=faiss_dir,
        config_path=config_path,
        created_at=created_at.isoformat(),
    )
    stamp = created_at.strftime("%Y%m%d-%H%M%S")
    filename = f"cv-arxiv-backup-{stamp}.tar.gz"
    response = Response(archive, mimetype="application/gzip")
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@api_bp.route("/backup/import", methods=["POST"])
def backup_import():
    # Raise the body-size limit before any form/file parsing (validate_csrf_token may
    # read request.form, which triggers it). Flask 3.1+ honours a per-request override.
    request.max_content_length = _MAX_BACKUP_UPLOAD_BYTES
    validate_csrf_token()
    uploaded = request.files.get("backup")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "No backup file uploaded"}), 400

    db_path, faiss_dir, config_path = _resolve_paths()
    try:
        summary = restore_backup(
            uploaded.read(),
            db_path=db_path,
            faiss_dir=faiss_dir,
            config_path=config_path,
        )
    except ValueError as exc:
        # Malformed / unsupported / oversized archive — a client error.
        return jsonify({"error": str(exc)}), 400
    except OSError as exc:
        # I/O failure during restore (disk full, permissions, read-only fs). The
        # restore rolls back on its own; surface a clean error instead of a 500.
        current_app.logger.exception("Backup restore failed with an OS error")
        return jsonify({"error": f"Restore failed: {exc}"}), 500

    # Best-effort reset of the embedding singleton so the freshly restored FAISS
    # index is reloaded without a full process restart. The reload is still only
    # partial in a running worker, hence the restart note in the response.
    try:
        from app.services.embeddings import reset_embedding_service

        reset_embedding_service()
    except Exception:  # pragma: no cover - reset is best-effort
        current_app.logger.exception("Failed to reset embedding service after restore")

    return jsonify(summary)
