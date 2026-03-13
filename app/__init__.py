"""Flask application factory."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from flask import Flask

from app.models import db
from app.schema import ensure_schema

DEFAULT_DATABASE_URI = "sqlite:///arxiv_papers.db"


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _validate_config(config: dict) -> None:
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

    if "SCRAPER_CONFIG" not in app.config:
        app.config["SCRAPER_CONFIG"] = _load_config(config_path)

    _validate_config(app.config["SCRAPER_CONFIG"])

    db.init_app(app)
    with app.app_context():
        db.create_all()
        ensure_schema()

    _register_blueprints(app)
    return app
