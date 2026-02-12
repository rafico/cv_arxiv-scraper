from pathlib import Path

import yaml
from flask import Flask

from app.models import db

DEFAULT_DATABASE_URI = "sqlite:///arxiv_papers.db"
DEFAULT_SECRET_KEY = "dev"


def _load_config(path):
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _register_blueprints(app):
    from app.routes.api import api_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.settings import settings_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(api_bp)


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.update(
        SQLALCHEMY_DATABASE_URI=DEFAULT_DATABASE_URI,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SECRET_KEY=DEFAULT_SECRET_KEY,
    )

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    config_path = (Path(app.root_path).parent / "config.yaml").resolve()
    app.config["CONFIG_PATH"] = str(config_path)
    app.config["SCRAPER_CONFIG"] = _load_config(config_path)

    db.init_app(app)
    with app.app_context():
        db.create_all()

    _register_blueprints(app)
    return app
