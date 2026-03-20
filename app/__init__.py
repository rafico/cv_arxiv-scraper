"""Flask application factory."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from flask import Flask

from app.models import db
from app.schema import ensure_schema
from app.services.preferences import get_preferences

DEFAULT_DATABASE_URI = "sqlite:///arxiv_papers.db"
DEFAULT_LLM_KEY_FILENAME = ".llm_api_key"


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _resolve_llm_key_path(config_path: Path | None = None) -> Path:
    base_dir = config_path.parent if config_path is not None else Path(__file__).resolve().parent.parent
    return base_dir / DEFAULT_LLM_KEY_FILENAME


def _llm_api_key_available(config_path: Path | None = None) -> bool:
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return True

    key_path = _resolve_llm_key_path(config_path)
    if not key_path.is_file():
        return False

    try:
        return bool(key_path.read_text(encoding="utf-8").strip())
    except OSError:
        return False


def _validate_config(config: dict, *, config_path: Path | None = None) -> None:
    if not isinstance(config, dict):
        raise ValueError("Config must be a dict")

    # --- scraper section ---
    if "scraper" not in config:
        raise ValueError("Missing required config section: 'scraper'")
    scraper = config["scraper"]
    if not isinstance(scraper, dict):
        raise ValueError("'scraper' must be a dict")
    feed_url = scraper.get("feed_url")
    if not isinstance(feed_url, str) or not feed_url.strip():
        raise ValueError("'scraper.feed_url' must be a non-empty string")

    # --- whitelists section ---
    if "whitelists" not in config:
        raise ValueError("Missing required config section: 'whitelists'")
    whitelists = config["whitelists"]
    if not isinstance(whitelists, dict):
        raise ValueError("'whitelists' must be a dict")
    for key in ("titles", "authors", "affiliations"):
        if key not in whitelists:
            raise ValueError(f"Missing key in 'whitelists': '{key}'")
        value = whitelists[key]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"'whitelists.{key}' must be a list of strings")

    preferences = config.get("preferences")
    if preferences is not None and not isinstance(preferences, dict):
        raise ValueError("'preferences' must be a dict")
    normalized_preferences = get_preferences(config)
    for key, value in normalized_preferences["ranking"].items():
        if value <= 0:
            raise ValueError(f"'preferences.ranking.{key}' must be positive")
    for key, items in normalized_preferences["muted"].items():
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise ValueError(f"'preferences.muted.{key}' must be a list of strings")

    llm = config.get("llm")
    if llm is None:
        return
    if not isinstance(llm, dict):
        raise ValueError("'llm' must be a dict")

    for key in ("model", "base_url"):
        value = llm.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"'llm.{key}' must be a non-empty string when provided")

    max_concurrent = llm.get("max_concurrent", 4)
    try:
        if int(max_concurrent) < 1:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("'llm.max_concurrent' must be a positive integer") from None

    provider = llm.get("provider", "openrouter")
    if provider not in ("openrouter", "ollama"):
        raise ValueError(f"'llm.provider' must be 'openrouter' or 'ollama', got '{provider}'")

    if llm.get("enabled") and provider != "ollama" and not _llm_api_key_available(config_path):
        raise ValueError(
            "LLM is enabled but no API key was found via OPENROUTER_API_KEY or .llm_api_key"
        )


def _register_blueprints(app: Flask) -> None:
    from app.routes.api import api_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.settings import settings_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(api_bp)


def create_app(config_overrides: dict | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config.update(
        SQLALCHEMY_DATABASE_URI=DEFAULT_DATABASE_URI,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SECRET_KEY=os.environ.get("SECRET_KEY") or os.urandom(32).hex(),
    )

    if config_overrides:
        app.config.update(config_overrides)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    config_path = Path(
        app.config.get("CONFIG_PATH", (Path(app.root_path).parent / "config.yaml").resolve())
    )
    app.config["CONFIG_PATH"] = str(config_path)
    app.config["LLM_KEY_PATH"] = str(
        Path(app.config.get("LLM_KEY_PATH", _resolve_llm_key_path(config_path))).resolve()
    )

    if "SCRAPER_CONFIG" not in app.config:
        app.config["SCRAPER_CONFIG"] = _load_config(config_path)

    _validate_config(app.config["SCRAPER_CONFIG"], config_path=config_path)

    db.init_app(app)
    with app.app_context():
        db.create_all()
        ensure_schema()

    _register_blueprints(app)
    return app
