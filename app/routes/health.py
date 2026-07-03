"""Unauthenticated liveness/readiness probe at ``/healthz``.

Localhost single-user app: no auth, cheap checks only. Lives at the top level
(not under ``/api``) by piggy-backing on the api blueprint's registration event
(see ``app/routes/api/__init__.py``), so the app factory needs no edit.

Returns ``{status, version, db_ok, faiss_ok, paper_count}``. HTTP 200 when the
core subsystems are healthy, 503 when degraded, so container/orchestrator health
checks can gate on it while the JSON body carries the detail.
"""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from app._version import __version__

health_bp = Blueprint("health", __name__)


def _db_status() -> tuple[bool, int]:
    """Return ``(db_ok, paper_count)``; never raises."""
    try:
        from app.models import Paper

        return True, int(Paper.query.count())
    except Exception:  # noqa: BLE001 — a probe must degrade, not 500
        return False, 0


def _faiss_ok() -> bool:
    """Whether the FAISS index loads. Cheap: does not load the embedding model."""
    try:
        from app.services.embeddings import get_embedding_service

        # ``index_size()`` reads only the on-disk FAISS index (an empty index on a
        # fresh install counts as healthy); it never triggers the SPECTER2 load.
        get_embedding_service(current_app._get_current_object()).index_size()
        return True
    except Exception:  # noqa: BLE001 — degrade instead of failing the probe
        return False


@health_bp.route("/healthz")
def healthz():
    db_ok, paper_count = _db_status()
    faiss_ok = _faiss_ok()
    status = "ok" if (db_ok and faiss_ok) else "degraded"
    payload = {
        "status": status,
        "version": __version__,
        "db_ok": db_ok,
        "faiss_ok": faiss_ok,
        "paper_count": paper_count,
    }
    return jsonify(payload), (200 if status == "ok" else 503)
