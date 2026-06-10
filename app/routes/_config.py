"""Shared helpers for routes that persist config.yaml changes."""

from __future__ import annotations

from pathlib import Path

from flask import current_app

from app import _validate_config
from app.services.preferences import save_config


def activate_saved_config(full_config: dict) -> None:
    """Swap a just-saved config into the running app."""
    current_app.config["SCRAPER_CONFIG"] = full_config
    current_app.config["USING_DEFAULT_CONFIG"] = False


def persist_config(full_config: dict) -> None:
    """Validate full_config, write it to config.yaml, and activate it.

    Raises ValueError when validation fails (nothing is written).
    """
    config_path = Path(current_app.config["CONFIG_PATH"])
    _validate_config(full_config, config_path=config_path)
    save_config(config_path, full_config)
    activate_saved_config(full_config)
