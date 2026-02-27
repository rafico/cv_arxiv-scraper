from __future__ import annotations

from pathlib import Path
from secrets import compare_digest, token_urlsafe

import yaml
from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

settings_bp = Blueprint("settings", __name__)
_CSRF_SESSION_KEY = "settings_csrf_token"


def _normalize_multiline(value: str) -> list[str]:
    entries = [line.strip() for line in value.splitlines() if line.strip()]
    return list(dict.fromkeys(entries))


def _get_or_create_csrf_token() -> str:
    csrf_token = session.get(_CSRF_SESSION_KEY)
    if csrf_token:
        return csrf_token

    csrf_token = token_urlsafe(32)
    session[_CSRF_SESSION_KEY] = csrf_token
    return csrf_token


def _validate_csrf_token() -> None:
    submitted_token = request.form.get("csrf_token", "")
    expected_token = session.get(_CSRF_SESSION_KEY, "")
    if not submitted_token or not expected_token or not compare_digest(submitted_token, expected_token):
        abort(400, description="Invalid CSRF token")


@settings_bp.route("/settings", methods=["GET"])
def view_settings():
    config = current_app.config["SCRAPER_CONFIG"]
    return render_template(
        "settings.html",
        whitelists=config["whitelists"],
        csrf_token=_get_or_create_csrf_token(),
    )


@settings_bp.route("/settings", methods=["POST"])
def save_settings():
    _validate_csrf_token()

    config_path = Path(current_app.config["CONFIG_PATH"])
    new_whitelists = {
        "titles": _normalize_multiline(request.form["titles"]),
        "affiliations": _normalize_multiline(request.form["affiliations"]),
        "authors": _normalize_multiline(request.form["authors"]),
    }

    with config_path.open("r", encoding="utf-8") as handle:
        full_config = yaml.safe_load(handle)

    full_config["whitelists"] = new_whitelists

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(full_config, handle, default_flow_style=False, sort_keys=False)

    current_app.config["SCRAPER_CONFIG"] = full_config
    session[_CSRF_SESSION_KEY] = token_urlsafe(32)
    flash("Settings saved successfully.", "success")
    return redirect(url_for("settings.view_settings"))
