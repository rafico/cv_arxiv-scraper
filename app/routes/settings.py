from __future__ import annotations

import logging
from pathlib import Path
from secrets import compare_digest, token_urlsafe

import yaml
from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

log = logging.getLogger(__name__)

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


def _save_config_key(key: str, value) -> None:
    """Update a top-level key in config.yaml and in-memory config."""
    config_path = Path(current_app.config["CONFIG_PATH"])

    with config_path.open("r", encoding="utf-8") as handle:
        full_config = yaml.safe_load(handle)

    full_config[key] = value

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(full_config, handle, default_flow_style=False, sort_keys=False)

    current_app.config["SCRAPER_CONFIG"] = full_config


@settings_bp.route("/settings", methods=["GET"])
def view_settings():
    from app.services.email_digest import check_gmail_auth_status

    config = current_app.config["SCRAPER_CONFIG"]
    email_cfg = config.get("email", {})
    gmail_status = check_gmail_auth_status()

    return render_template(
        "settings.html",
        whitelists=config["whitelists"],
        email_config={
            "recipient": email_cfg.get("recipient", ""),
            "subject_prefix": email_cfg.get("subject_prefix", "ArXiv Digest"),
        },
        gmail_status=gmail_status,
        csrf_token=_get_or_create_csrf_token(),
        callback_uri=url_for("settings.gmail_callback", _external=True),
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


# ── Gmail / Email API endpoints ─────────────────────────────────────────


@settings_bp.route("/settings/gmail-status", methods=["GET"])
def gmail_status():
    from app.services.email_digest import check_gmail_auth_status

    return jsonify(check_gmail_auth_status())


@settings_bp.route("/settings/email", methods=["POST"])
def save_email_settings():
    _validate_csrf_token()

    recipient = request.form.get("email_recipient", "").strip()
    subject_prefix = request.form.get("email_subject_prefix", "ArXiv Digest").strip()

    email_cfg = {"recipient": recipient, "subject_prefix": subject_prefix}
    _save_config_key("email", email_cfg)

    session[_CSRF_SESSION_KEY] = token_urlsafe(32)
    flash("Email settings saved.", "success")
    return redirect(url_for("settings.view_settings"))


@settings_bp.route("/settings/gmail-auth", methods=["POST"])
def gmail_auth():
    _validate_csrf_token()

    from app.services.email_digest import start_oauth_flow

    callback_url = url_for("settings.gmail_callback", _external=True)
    result = start_oauth_flow(redirect_uri=callback_url)

    if not result["success"]:
        flash(result["message"], "error")
        return redirect(url_for("settings.view_settings"))

    session["oauth_state"] = result["state"]
    return redirect(result["auth_url"])


@settings_bp.route("/settings/gmail-callback", methods=["GET"])
def gmail_callback():
    from app.services.email_digest import finish_oauth_flow

    # Verify state to prevent CSRF on the OAuth callback.
    stored_state = session.pop("oauth_state", None)
    returned_state = request.args.get("state", "")
    if not stored_state or stored_state != returned_state:
        flash("OAuth state mismatch — please try authorizing again.", "error")
        return redirect(url_for("settings.view_settings"))

    error = request.args.get("error")
    if error:
        flash(f"Google authorization denied: {error}", "error")
        return redirect(url_for("settings.view_settings"))

    callback_url = url_for("settings.gmail_callback", _external=True)
    result = finish_oauth_flow(
        authorization_response_url=request.url,
        redirect_uri=callback_url,
    )

    if result["success"]:
        flash(result["message"], "success")
    else:
        flash(result["message"], "error")

    return redirect(url_for("settings.view_settings"))


@settings_bp.route("/settings/send-test-digest", methods=["POST"])
def send_test_digest():
    _validate_csrf_token()

    from app.services.email_digest import send_digest

    try:
        info = send_digest(current_app._get_current_object())
        flash(
            f"Test digest sent to {info['recipient']} ({info['papers_count']} papers).",
            "success",
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        flash(f"Failed to send digest: {exc}", "error")
        log.exception("Test digest failed")

    return redirect(url_for("settings.view_settings"))

