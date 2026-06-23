"""Shared helpers for routes that persist config.yaml changes."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

from flask import current_app

from app import _validate_config
from app.services.preferences import save_config

# Serializes the whole read-modify-write cycle done by mutating settings routes.
# Re-entrant so a route may call a helper that also takes it (e.g. _save_config_key).
_CONFIG_WRITE_LOCK = threading.RLock()


@contextmanager
def config_write_lock():
    """Serialize a full config read-modify-write across settings routes.

    ``save_config`` makes the file *write* atomic, but each mutating route does
    load -> mutate one section -> persist. Without serialization, two concurrent
    edits to different sections both persist on top of the same stale snapshot,
    silently clobbering one another. Hold this around the whole load/mutate/persist
    cycle so the loser re-reads the winner's write first.
    """
    with _CONFIG_WRITE_LOCK:
        yield


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
