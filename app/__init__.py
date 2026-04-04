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
_SECRET_KEY_FILENAME = ".flask_secret"


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
    feed_urls = scraper.get("feed_urls")
    feed_url = scraper.get("feed_url")
    if not feed_urls and not feed_url:
        raise ValueError("Must provide either 'scraper.feed_urls' or 'scraper.feed_url'")
    if feed_urls and (not isinstance(feed_urls, list) or not all(isinstance(u, str) and u.strip() for u in feed_urls)):
        raise ValueError("'scraper.feed_urls' must be a list of non-empty strings")
    if feed_url and (not isinstance(feed_url, str) or not feed_url.strip()):
        raise ValueError("'scraper.feed_url' must be a non-empty string")

    ingest = config.get("ingest")
    if ingest is not None:
        if not isinstance(ingest, dict):
            raise ValueError("'ingest' must be a dict")

        backends = ingest.get("backends")
        if backends is not None:
            if not isinstance(backends, list) or not backends or not all(isinstance(name, str) and name.strip() for name in backends):
                raise ValueError("'ingest.backends' must be a non-empty list of backend names")

            from app.services.ingest.orchestrator import BACKEND_REGISTRY

            unknown_backends = [name for name in backends if name not in BACKEND_REGISTRY]
            if unknown_backends:
                raise ValueError(
                    f"'ingest.backends' contains unknown backends: {', '.join(unknown_backends)}"
                )

        user_agent = ingest.get("user_agent")
        if user_agent is not None and (not isinstance(user_agent, str) or not user_agent.strip()):
            raise ValueError("'ingest.user_agent' must be a non-empty string when provided")

        rate_limit = ingest.get("rate_limit")
        if rate_limit is not None:
            if not isinstance(rate_limit, dict):
                raise ValueError("'ingest.rate_limit' must be a dict")

            requests_per_second = rate_limit.get("requests_per_second")
            if requests_per_second is not None:
                if isinstance(requests_per_second, bool):
                    raise ValueError("'ingest.rate_limit.requests_per_second' must be positive")
                try:
                    if float(requests_per_second) <= 0:
                        raise ValueError
                except (TypeError, ValueError):
                    raise ValueError("'ingest.rate_limit.requests_per_second' must be positive") from None

            burst = rate_limit.get("burst")
            if burst is not None:
                if isinstance(burst, bool):
                    raise ValueError("'ingest.rate_limit.burst' must be a positive integer")
                try:
                    burst_value = int(burst)
                    if float(burst_value) != float(burst) or burst_value < 1:
                        raise ValueError
                except (TypeError, ValueError):
                    raise ValueError("'ingest.rate_limit.burst' must be a positive integer") from None

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
        if value > 1000:
            raise ValueError(f"'preferences.ranking.{key}' must be at most 1000")
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
        raise ValueError("LLM is enabled but no API key was found via OPENROUTER_API_KEY or .llm_api_key")


def _register_blueprints(app: Flask) -> None:
    from app.routes.api import api_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.discover import discover_bp
    from app.routes.help import help_bp
    from app.routes.settings import settings_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(discover_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(help_bp)
    app.register_blueprint(api_bp)


def _ensure_secret_key(instance_path: Path) -> str:
    """Load or generate a persistent secret key in the instance folder."""
    env_key = os.environ.get("SECRET_KEY", "").strip()
    if env_key:
        return env_key
    key_path = instance_path / _SECRET_KEY_FILENAME
    if key_path.is_file():
        try:
            return key_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    key = os.urandom(32).hex()
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(key, encoding="utf-8")
        os.chmod(key_path, 0o600)
    except OSError:
        pass  # Fall back to ephemeral key if we can't write.
    return key


def create_app(config_overrides: dict | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    app.config.update(
        SQLALCHEMY_DATABASE_URI=DEFAULT_DATABASE_URI,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SECRET_KEY=_ensure_secret_key(Path(app.instance_path)),
    )

    if config_overrides:
        app.config.update(config_overrides)

    # instance_path already created above before _ensure_secret_key call
    app.config.setdefault("FAISS_INDEX_DIR", str(Path(app.instance_path) / "faiss_index"))

    config_path = Path(app.config.get("CONFIG_PATH", (Path(app.root_path).parent / "config.yaml").resolve()))
    app.config["CONFIG_PATH"] = str(config_path)
    app.config["LLM_KEY_PATH"] = str(Path(app.config.get("LLM_KEY_PATH", _resolve_llm_key_path(config_path))).resolve())

    if "SCRAPER_CONFIG" not in app.config:
        app.config["SCRAPER_CONFIG"] = _load_config(config_path)

    _validate_config(app.config["SCRAPER_CONFIG"], config_path=config_path)

    db.init_app(app)
    with app.app_context():
        db.create_all()
        ensure_schema()

    _register_blueprints(app)

    from app.constants import ARXIV_CATEGORY_NAMES

    app.jinja_env.globals["ARXIV_CATEGORY_NAMES"] = ARXIV_CATEGORY_NAMES

    # Start built-in scheduler if configured.
    scheduler_config = app.config["SCRAPER_CONFIG"].get("scheduler", {})
    if scheduler_config.get("enabled"):
        from app.services.scheduler import SCRAPE_SCHEDULER

        daily_at = str(scheduler_config.get("daily_at", "08:00"))
        SCRAPE_SCHEDULER.start(app, daily_at=daily_at)

    return app
