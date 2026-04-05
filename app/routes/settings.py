from __future__ import annotations

import logging
from pathlib import Path
from secrets import token_urlsafe

import yaml
from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from app import _validate_config
from app.constants import DEFAULT_LLM_MODEL
from app.csrf import get_or_create_csrf_token, validate_csrf_token
from app.models import Paper, db
from app.services.llm_client import has_api_key, write_api_key
from app.services.preferences import get_preferences, save_config, update_preferences_from_form
from app.services.ranking import recompute_all_paper_scores

LOGGER = logging.getLogger(__name__)

settings_bp = Blueprint("settings", __name__)
_LLM_MASK_VALUE = "********"


def _normalize_multiline(value: str) -> list[str]:
    entries = [line.strip() for line in value.splitlines() if line.strip()]
    return list(dict.fromkeys(entries))


def _save_config_key(key: str, value) -> None:
    """Update a top-level key in config.yaml and in-memory config."""
    config_path = Path(current_app.config["CONFIG_PATH"])

    with config_path.open("r", encoding="utf-8") as handle:
        full_config = yaml.safe_load(handle)

    full_config[key] = value

    save_config(config_path, full_config)
    current_app.config["SCRAPER_CONFIG"] = full_config


def _load_full_config() -> dict:
    config_path = Path(current_app.config["CONFIG_PATH"])
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _llm_provider_defaults(provider: str) -> dict[str, str]:
    if provider == "ollama":
        return {
            "model": "llama3",
            "base_url": "http://localhost:11434/v1",
        }
    return {
        "model": DEFAULT_LLM_MODEL,
        "base_url": "https://openrouter.ai/api/v1",
    }


def _build_llm_view_model(config: dict) -> dict:
    llm_cfg = config.get("llm", {})
    provider = llm_cfg.get("provider", "openrouter")
    defaults = _llm_provider_defaults(provider)
    return {
        "enabled": bool(llm_cfg.get("enabled", False)),
        "provider": provider,
        "model": llm_cfg.get("model", defaults["model"]),
        "base_url": llm_cfg.get("base_url", defaults["base_url"]),
        "max_concurrent": int(llm_cfg.get("max_concurrent", 4) or 4),
    }


@settings_bp.route("/settings", methods=["GET"])
def view_settings():
    from app.models import ScrapeRun
    from app.services.email_digest import (
        build_digest_preview,
        check_gmail_auth_status,
        get_digest_history,
        get_digest_status_snapshot,
        get_setup_instructions,
        validate_credentials_redirect_uris,
    )

    config = current_app.config["SCRAPER_CONFIG"]
    email_cfg = config.get("email", {})
    gmail_status = check_gmail_auth_status()
    llm_key_path = Path(current_app.config["LLM_KEY_PATH"])
    section = request.args.get("section", "interests")
    preferences = get_preferences(config)
    digest_status = get_digest_status_snapshot(current_app._get_current_object())
    digest_preview = build_digest_preview(current_app._get_current_object())
    scrape_history = ScrapeRun.query.order_by(ScrapeRun.started_at.desc()).limit(8).all()
    digest_history = get_digest_history(limit=8)
    callback_uri = url_for("settings.gmail_callback", _external=True)
    gmail_setup_steps = get_setup_instructions(
        callback_uri=callback_uri,
        recipient=email_cfg.get("recipient", ""),
    )
    redirect_uri_check = validate_credentials_redirect_uris(callback_uri)

    from app.services.cron import get_cron_status
    from app.services.mendeley import MendeleyClient
    from app.services.zotero import ZoteroClient

    mendeley_status = MendeleyClient().check_connection()
    zotero_client = ZoteroClient()
    zotero_status = zotero_client.check_connection()
    zotero_collections = zotero_client.list_collections() if zotero_status["status"] == "connected" else []
    cron_config = get_cron_status()

    return render_template(
        "settings.html",
        section=section,
        whitelists=config["whitelists"],
        preferences=preferences,
        email_config={
            "recipient": email_cfg.get("recipient", ""),
            "subject_prefix": email_cfg.get("subject_prefix", "ArXiv Digest"),
        },
        gmail_status=gmail_status,
        gmail_setup_steps=gmail_setup_steps,
        redirect_uri_check=redirect_uri_check,
        digest_status=digest_status,
        digest_preview=digest_preview,
        digest_history=digest_history,
        scrape_history=scrape_history,
        llm_config=_build_llm_view_model(config),
        llm_key_configured=has_api_key(llm_key_path),
        llm_key_mask=_LLM_MASK_VALUE,
        csrf_token=get_or_create_csrf_token(),
        callback_uri=callback_uri,
        mendeley_status=mendeley_status,
        zotero_status=zotero_status,
        zotero_collections=zotero_collections,
        cron_config=cron_config,
    )


@settings_bp.route("/settings", methods=["POST"])
def save_settings():
    validate_csrf_token()

    config_path = Path(current_app.config["CONFIG_PATH"])
    new_whitelists = {
        "titles": _normalize_multiline(request.form.get("titles", "")),
        "affiliations": _normalize_multiline(request.form.get("affiliations", "")),
        "authors": _normalize_multiline(request.form.get("authors", "")),
    }

    with config_path.open("r", encoding="utf-8") as handle:
        full_config = yaml.safe_load(handle)

    full_config["whitelists"] = new_whitelists

    try:
        _validate_config(full_config, config_path=config_path)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("settings.view_settings", section="interests"))

    save_config(config_path, full_config)
    current_app.config["SCRAPER_CONFIG"] = full_config
    session["settings_csrf_token"] = token_urlsafe(32)
    flash("Settings saved successfully.", "success")
    return redirect(url_for("settings.view_settings", section="interests"))


@settings_bp.route("/settings/preferences", methods=["POST"])
def save_preferences():
    validate_csrf_token()

    config_path = Path(current_app.config["CONFIG_PATH"])
    try:
        full_config = update_preferences_from_form(_load_full_config(), request.form)
        _validate_config(full_config, config_path=config_path)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("settings.view_settings", section="controls"))

    save_config(config_path, full_config)
    current_app.config["SCRAPER_CONFIG"] = full_config
    recompute_all_paper_scores(current_app._get_current_object())
    session["settings_csrf_token"] = token_urlsafe(32)
    flash("Ranking and mute preferences saved.", "success")
    return redirect(url_for("settings.view_settings", section="controls"))


# ── Gmail / Email API endpoints ─────────────────────────────────────────


@settings_bp.route("/settings/upload-credentials", methods=["POST"])
def upload_credentials():
    """Accept a credentials.json file upload and save to project root."""
    import json as _json

    validate_csrf_token()

    uploaded = request.files.get("credentials_file")
    if not uploaded or not uploaded.filename:
        flash("No file selected.", "error")
        return redirect(url_for("settings.view_settings", section="automation"))

    try:
        raw = uploaded.read()
        data = _json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        flash("Invalid JSON file. Please upload the credentials.json from Google Cloud Console.", "error")
        return redirect(url_for("settings.view_settings", section="automation"))

    # Validate expected OAuth client structure.
    web = data.get("web", {})
    if not web.get("client_id") or not web.get("client_secret"):
        flash(
            "This file doesn't look like a valid OAuth credentials file. "
            "Make sure you download the JSON for an OAuth 2.0 Client ID "
            "(Web application type).",
            "error",
        )
        return redirect(url_for("settings.view_settings", section="automation"))

    from app.services.email_digest import DEFAULT_CREDENTIALS_PATH, validate_credentials_redirect_uris

    DEFAULT_CREDENTIALS_PATH.write_bytes(raw)

    callback_uri = url_for("settings.gmail_callback", _external=True)
    uri_check = validate_credentials_redirect_uris(callback_uri)
    if not uri_check["match"]:
        flash(
            f"credentials.json uploaded. Warning: {uri_check['message']}",
            "success",
        )
    else:
        flash("credentials.json uploaded successfully. You can now authorize Gmail.", "success")
    return redirect(url_for("settings.view_settings", section="automation"))


@settings_bp.route("/settings/gmail-status", methods=["GET"])
def gmail_status():
    from app.services.email_digest import check_gmail_auth_status

    return jsonify(check_gmail_auth_status())


@settings_bp.route("/settings/email", methods=["POST"])
def save_email_settings():
    validate_csrf_token()

    recipient = request.form.get("email_recipient", "").strip()
    subject_prefix = request.form.get("email_subject_prefix", "ArXiv Digest").strip()

    email_cfg = {"recipient": recipient, "subject_prefix": subject_prefix}
    _save_config_key("email", email_cfg)

    session["settings_csrf_token"] = token_urlsafe(32)
    flash("Email settings saved.", "success")
    return redirect(url_for("settings.view_settings", section="automation"))


@settings_bp.route("/settings/llm", methods=["POST"])
def save_llm_settings():
    validate_csrf_token()

    config_path = Path(current_app.config["CONFIG_PATH"])
    key_path = Path(current_app.config["LLM_KEY_PATH"])
    enabled = request.form.get("llm_enabled") == "on"
    provider = request.form.get("llm_provider", "openrouter").strip()
    if provider not in ("openrouter", "ollama"):
        provider = "openrouter"
    defaults = _llm_provider_defaults(provider)
    model = request.form.get("llm_model", "").strip()
    base_url = request.form.get("llm_base_url", "").strip()
    max_concurrent_raw = request.form.get("llm_max_concurrent", "4").strip()
    api_key = request.form.get("llm_api_key", "").strip()

    model = model or defaults["model"]
    base_url = base_url or defaults["base_url"]

    try:
        max_concurrent = max(1, int(max_concurrent_raw or "4"))
    except ValueError:
        flash("LLM max concurrent requests must be a positive integer.", "error")
        return redirect(url_for("settings.view_settings", section="ai"))

    with config_path.open("r", encoding="utf-8") as handle:
        full_config = yaml.safe_load(handle)

    full_config["llm"] = {
        "enabled": enabled,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "max_concurrent": max_concurrent,
    }

    if provider != "ollama" and api_key and api_key != _LLM_MASK_VALUE:
        write_api_key(api_key, key_path)

    try:
        _validate_config(full_config, config_path=config_path)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("settings.view_settings", section="ai"))

    save_config(config_path, full_config)
    current_app.config["SCRAPER_CONFIG"] = full_config
    session["settings_csrf_token"] = token_urlsafe(32)
    flash("LLM settings saved.", "success")
    return redirect(url_for("settings.view_settings", section="ai"))


@settings_bp.route("/settings/gmail-auth", methods=["POST"])
def gmail_auth():
    validate_csrf_token()

    from app.services.email_digest import start_oauth_flow

    callback_url = url_for("settings.gmail_callback", _external=True)
    result = start_oauth_flow(redirect_uri=callback_url)

    if not result["success"]:
        flash(result["message"], "error")
        return redirect(url_for("settings.view_settings", section="automation"))

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
        return redirect(url_for("settings.view_settings", section="automation"))

    error = request.args.get("error")
    if error:
        flash(f"Google authorization denied: {error}", "error")
        return redirect(url_for("settings.view_settings", section="automation"))

    callback_url = url_for("settings.gmail_callback", _external=True)
    result = finish_oauth_flow(
        authorization_response_url=request.url,
        redirect_uri=callback_url,
    )

    if result["success"]:
        flash(result["message"], "success")
    else:
        flash(result["message"], "error")

    return redirect(url_for("settings.view_settings", section="automation"))


@settings_bp.route("/settings/send-test-digest", methods=["POST"])
def send_test_digest():
    validate_csrf_token()

    from app.services.email_digest import send_digest

    try:
        info = send_digest(current_app._get_current_object())
        flash(
            f"Test digest sent to {info['recipient']} ({info['papers_count']} papers).",
            "success",
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        flash(f"Failed to send digest: {exc}", "error")
        LOGGER.exception("Test digest failed")

    return redirect(url_for("settings.view_settings", section="automation"))


@settings_bp.route("/settings/digest-preview", methods=["GET"])
def digest_preview():
    from app.services.email_digest import build_digest_preview

    preview = build_digest_preview(current_app._get_current_object())
    response = current_app.response_class(preview["html"], mimetype="text/html")
    response.headers["X-Digest-Subject"] = preview["subject"]
    return response


# ── Cron scheduling endpoint ─────────────────────────────────────────


@settings_bp.route("/settings/cron", methods=["POST"])
def manage_cron():
    validate_csrf_token()

    from app.services.cron import install_cron_job, remove_cron_job

    action = request.form.get("cron_action", "install")

    if action == "remove":
        result = remove_cron_job()
    else:
        try:
            hour = int(request.form.get("cron_hour", 8))
            minute = int(request.form.get("cron_minute", 0))
        except (ValueError, TypeError):
            flash("Hour and minute must be integers.", "error")
            return redirect(url_for("settings.view_settings", section="automation"))
        mode = request.form.get("cron_mode", "full")
        result = install_cron_job(hour, minute, mode)

    flash(result["message"], "success" if result["success"] else "error")
    return redirect(url_for("settings.view_settings", section="automation"))


# ── Mendeley endpoints ───────────────────────────────────────────────


@settings_bp.route("/settings/upload-mendeley-credentials", methods=["POST"])
def upload_mendeley_credentials():
    """Accept a mendeley_credentials.json file upload."""
    import json as _json

    validate_csrf_token()

    uploaded = request.files.get("mendeley_credentials_file")
    if not uploaded or not uploaded.filename:
        flash("No file selected.", "error")
        return redirect(url_for("settings.view_settings", section="automation"))

    try:
        raw = uploaded.read()
        data = _json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        flash("Invalid JSON file.", "error")
        return redirect(url_for("settings.view_settings", section="automation"))

    if not data.get("client_id") or not data.get("client_secret"):
        flash(
            "Missing client_id or client_secret. Upload the credentials JSON from Mendeley developer portal.",
            "error",
        )
        return redirect(url_for("settings.view_settings", section="automation"))

    from app.services.mendeley import DEFAULT_CREDENTIALS_PATH

    DEFAULT_CREDENTIALS_PATH.write_bytes(raw)
    flash("Mendeley credentials uploaded. You can now authorize.", "success")
    return redirect(url_for("settings.view_settings", section="automation"))


@settings_bp.route("/settings/mendeley-auth", methods=["POST"])
def mendeley_auth():
    validate_csrf_token()

    from app.services.mendeley import MendeleyClient

    client = MendeleyClient()
    callback_url = url_for("settings.mendeley_callback", _external=True)
    result = client.start_oauth_flow(redirect_uri=callback_url)

    if not result["success"]:
        flash(result["message"], "error")
        return redirect(url_for("settings.view_settings", section="automation"))

    session["mendeley_oauth_state"] = result["state"]
    return redirect(result["auth_url"])


@settings_bp.route("/settings/mendeley-callback", methods=["GET"])
def mendeley_callback():
    from app.services.mendeley import MendeleyClient

    stored_state = session.pop("mendeley_oauth_state", None)
    returned_state = request.args.get("state", "")
    if not stored_state or stored_state != returned_state:
        flash("OAuth state mismatch -- please try authorizing again.", "error")
        return redirect(url_for("settings.view_settings", section="automation"))

    error = request.args.get("error")
    if error:
        flash(f"Mendeley authorization denied: {error}", "error")
        return redirect(url_for("settings.view_settings", section="automation"))

    client = MendeleyClient()
    callback_url = url_for("settings.mendeley_callback", _external=True)
    result = client.finish_oauth_flow(
        authorization_response_url=request.url,
        redirect_uri=callback_url,
    )

    flash(result["message"], "success" if result["success"] else "error")
    return redirect(url_for("settings.view_settings", section="automation"))


@settings_bp.route("/settings/mendeley-sync", methods=["POST"])
def mendeley_sync():
    validate_csrf_token()

    from app.models import PaperFeedback
    from app.services.mendeley import MendeleyClient

    client = MendeleyClient()
    status = client.check_connection()
    if status["status"] != "connected":
        flash(f"Mendeley not connected: {status['message']}", "error")
        return redirect(url_for("settings.view_settings", section="automation"))

    saved_papers = Paper.query.join(
        PaperFeedback,
        db.and_(
            PaperFeedback.paper_id == Paper.id,
            PaperFeedback.action == "save",
        ),
    ).all()

    success_count = 0
    for paper in saved_papers:
        result = client.add_document(paper)
        if result["success"]:
            success_count += 1

    flash(f"Synced {success_count}/{len(saved_papers)} papers to Mendeley.", "success")
    return redirect(url_for("settings.view_settings", section="automation"))


# ── Zotero endpoints ─────────────────────────────────────────────────


@settings_bp.route("/settings/zotero-setup", methods=["POST"])
def zotero_setup():
    validate_csrf_token()

    from app.services.zotero import ZoteroClient

    api_key = request.form.get("zotero_api_key", "").strip()
    user_id = request.form.get("zotero_user_id", "").strip()

    if not api_key or not user_id:
        flash("Both API key and user ID are required.", "error")
        return redirect(url_for("settings.view_settings", section="automation"))

    client = ZoteroClient()
    client._save_credentials(api_key, user_id)

    status = client.check_connection()
    if status["status"] == "connected":
        flash("Zotero connected successfully.", "success")
    else:
        flash(f"Credentials saved but connection failed: {status['message']}", "error")

    return redirect(url_for("settings.view_settings", section="automation"))


@settings_bp.route("/settings/zotero-test", methods=["POST"])
def zotero_test():
    validate_csrf_token()

    from app.services.zotero import ZoteroClient

    client = ZoteroClient()
    status = client.check_connection()
    flash(status["message"], "success" if status["status"] == "connected" else "error")
    return redirect(url_for("settings.view_settings", section="automation"))


@settings_bp.route("/settings/zotero-collections", methods=["GET"])
def zotero_collections():
    from app.services.zotero import ZoteroClient

    client = ZoteroClient()
    collections = client.list_collections()
    return jsonify(collections)


@settings_bp.route("/settings/zotero-sync", methods=["POST"])
def zotero_sync():
    validate_csrf_token()

    from app.models import PaperFeedback
    from app.services.zotero import ZoteroClient

    client = ZoteroClient()
    status = client.check_connection()
    if status["status"] != "connected":
        flash(f"Zotero not connected: {status['message']}", "error")
        return redirect(url_for("settings.view_settings", section="automation"))

    collection_key = request.form.get("zotero_collection", "").strip() or None

    saved_papers = Paper.query.join(
        PaperFeedback,
        db.and_(
            PaperFeedback.paper_id == Paper.id,
            PaperFeedback.action == "save",
        ),
    ).all()

    result = client.sync_saved_papers(saved_papers, collection_key=collection_key)
    flash(result["message"], "success" if result["success"] else "error")
    return redirect(url_for("settings.view_settings", section="automation"))
