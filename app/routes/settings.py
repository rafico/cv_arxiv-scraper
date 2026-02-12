import yaml
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    current_app,
)

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/settings", methods=["GET"])
def view_settings():
    config = current_app.config["SCRAPER_CONFIG"]
    return render_template("settings.html", whitelists=config["whitelists"])


@settings_bp.route("/settings", methods=["POST"])
def save_settings():
    config_path = current_app.config["CONFIG_PATH"]

    new_whitelists = {
        "titles": [
            t.strip() for t in request.form["titles"].splitlines() if t.strip()
        ],
        "affiliations": [
            a.strip()
            for a in request.form["affiliations"].splitlines()
            if a.strip()
        ],
        "authors": [
            a.strip() for a in request.form["authors"].splitlines() if a.strip()
        ],
    }

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    full_config["whitelists"] = new_whitelists

    with open(config_path, "w") as f:
        yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)

    current_app.config["SCRAPER_CONFIG"] = full_config
    flash("Settings saved successfully.", "success")
    return redirect(url_for("settings.view_settings"))
