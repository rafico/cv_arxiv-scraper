"""Interest-profile management endpoints (Wave 3).

CRUD for named interest profiles, each with its own learned ranker, feed, and
digest section plus an editable natural-language description. Mutations are CSRF
guarded and validated through the shared ``_validation`` helpers; the service
layer (``app.services.profiles``) enforces the invariants (exactly one active,
exactly one default, cannot delete the default or the last profile).
"""

from __future__ import annotations

import logging

from flask import current_app, jsonify, request

from app.csrf import validate_csrf_token
from app.routes.api import api_bp
from app.routes.api._validation import optional_str, require_str
from app.services import profiles as profiles_service

LOGGER = logging.getLogger(__name__)


def _rerank_active_profile() -> None:
    """Rebuild interest similarities + scores for the newly active profile.

    Best-effort: a ranking failure must never fail the profile switch itself.
    """
    try:
        from app.services.interest_model import build_interest_profile, recompute_interest_similarities

        app = current_app._get_current_object()
        if build_interest_profile(app) is not None:
            recompute_interest_similarities(app)
        else:
            from app.services.ranking import recompute_all_paper_scores

            recompute_all_paper_scores(app)
    except Exception:  # pragma: no cover - re-rank is best-effort
        LOGGER.warning("Re-rank after profile switch failed (non-fatal)", exc_info=True)


@api_bp.route("/profiles", methods=["GET"])
def list_profiles():
    profiles = profiles_service.list_profiles()
    return jsonify({"profiles": [profiles_service.profile_to_dict(p) for p in profiles]})


@api_bp.route("/profiles", methods=["POST"])
def create_profile():
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    name = require_str(payload, "name")
    description = optional_str(payload, "description")
    try:
        profile = profiles_service.create_profile(name, description)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(profiles_service.profile_to_dict(profile)), 201


@api_bp.route("/profiles/<int:profile_id>/rename", methods=["POST"])
def rename_profile(profile_id: int):
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    name = require_str(payload, "name")
    try:
        profile = profiles_service.rename_profile(profile_id, name)
    except LookupError:
        return jsonify({"error": "Profile not found"}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(profiles_service.profile_to_dict(profile))


@api_bp.route("/profiles/<int:profile_id>/description", methods=["POST"])
def update_description(profile_id: int):
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    description = optional_str(payload, "description")
    try:
        profile = profiles_service.update_description(profile_id, description)
    except LookupError:
        return jsonify({"error": "Profile not found"}), 404
    # A changed description alters cold-start scoring; refresh the feed if active.
    if profile.is_active:
        _rerank_active_profile()
    return jsonify(profiles_service.profile_to_dict(profile))


@api_bp.route("/profiles/<int:profile_id>/activate", methods=["POST"])
def activate_profile(profile_id: int):
    validate_csrf_token()
    try:
        profile = profiles_service.set_active_profile(profile_id)
    except LookupError:
        return jsonify({"error": "Profile not found"}), 404
    _rerank_active_profile()
    return jsonify(profiles_service.profile_to_dict(profile))


@api_bp.route("/profiles/<int:profile_id>/digest", methods=["POST"])
def toggle_digest(profile_id: int):
    validate_csrf_token()
    payload = request.get_json(silent=True) or {}
    from app import _is_truthy_flag

    value = _is_truthy_flag(payload.get("include_in_digest", True))
    try:
        profile = profiles_service.set_include_in_digest(profile_id, value)
    except LookupError:
        return jsonify({"error": "Profile not found"}), 404
    return jsonify(profiles_service.profile_to_dict(profile))


@api_bp.route("/profiles/<int:profile_id>", methods=["DELETE"])
def delete_profile(profile_id: int):
    validate_csrf_token()
    try:
        was_active = bool(profiles_service.get_profile(profile_id) and profiles_service.get_profile(profile_id).is_active)
        profiles_service.delete_profile(profile_id)
    except LookupError:
        return jsonify({"error": "Profile not found"}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if was_active:
        _rerank_active_profile()
    return jsonify({"deleted": True})
