"""Config-backed product preferences and lightweight recommendations."""

from __future__ import annotations

import os
import tempfile
import threading
from copy import deepcopy
from pathlib import Path

import yaml

# Module-level lock to serialize config read-modify-write cycles.
_CONFIG_LOCK = threading.Lock()

DEFAULT_PREFERENCES: dict[str, dict] = {
    "ranking": {
        "author_weight": 44.0,
        "affiliation_weight": 26.0,
        "title_weight": 14.0,
        "ai_weight": 5.0,
        "citation_weight": 0.05,
        "venue_weight": 8.0,
        "interest_weight": 12.0,
        "freshness_half_life_days": 14.0,
    },
    "display": {
        "summary_lines": 3,
        "show_score_breakdown_bars": True,
    },
    "muted": {
        "authors": [],
        "affiliations": [],
        "topics": [],
    },
}


def _dedupe_str_list(items: list[str]) -> list[str]:
    cleaned = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    return list(dict.fromkeys(cleaned))


def _as_str_list(value: object) -> list[str]:
    """Coerce a config value into a list of strings.

    A hand-edited ``config.yaml`` can store a whitelist as a bare YAML scalar
    (``authors: Jane Doe``) instead of a list. ``list("Jane Doe")`` would explode
    that into single characters, so treat a scalar string as a one-element list
    and anything that isn't a list as empty.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    return []


def get_preferences(config: dict | None) -> dict:
    merged = deepcopy(DEFAULT_PREFERENCES)
    if not isinstance(config, dict):
        return merged

    raw = config.get("preferences", {})
    if not isinstance(raw, dict):
        return merged

    ranking = raw.get("ranking", {})
    if isinstance(ranking, dict):
        for key, default_value in merged["ranking"].items():
            value = ranking.get(key)
            try:
                if value is not None:
                    merged["ranking"][key] = float(value)
            except (TypeError, ValueError):
                merged["ranking"][key] = float(default_value)

    display = raw.get("display", {})
    if isinstance(display, dict):
        for key, default_value in merged["display"].items():
            value = display.get(key)
            if isinstance(default_value, bool):
                merged["display"][key] = default_value if value is None else bool(value)
                continue
            try:
                if value is not None:
                    merged["display"][key] = max(1, min(10, int(value)))
            except (TypeError, ValueError):
                merged["display"][key] = int(default_value)

    muted = raw.get("muted", {})
    if isinstance(muted, dict):
        for key in merged["muted"]:
            merged["muted"][key] = _dedupe_str_list(_as_str_list(muted.get(key)))

    return merged


def update_preferences_from_form(config: dict, form) -> dict:
    updated = deepcopy(config)
    preferences = get_preferences(config)

    ranking = preferences["ranking"]
    for key in ranking:
        raw = form.get(f"pref_{key}", "").strip()
        if not raw:
            continue
        ranking[key] = float(raw)

    display = preferences["display"]
    for key in display:
        if isinstance(display[key], bool):
            # Checkbox: only present in the form payload when ticked.
            display[key] = f"display_{key}" in form
            continue
        raw = form.get(f"display_{key}", "").strip()
        if raw:
            try:
                display[key] = max(1, min(10, int(raw)))
            except (TypeError, ValueError):
                pass

    muted = preferences["muted"]
    for key in muted:
        raw = form.get(f"muted_{key}", "")
        muted[key] = _dedupe_str_list(raw.splitlines())

    updated["preferences"] = preferences
    return updated


def save_config(config_path: Path, full_config: dict) -> dict:
    """Write config to disk atomically (write-to-temp + rename).

    Falls back to an in-place write when the destination cannot be replaced by
    rename — e.g. a Docker single-file bind mount, where renaming over the mount
    point fails with EBUSY/EINVAL. Atomicity is preserved on the common path.
    """
    with _CONFIG_LOCK:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = yaml.safe_dump(full_config, default_flow_style=False, sort_keys=False)
        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
            try:
                os.replace(tmp_path, config_path)
            except OSError:
                # Destination is likely a bind-mounted file; rename can't replace
                # a mount point, so write through it in place instead.
                with open(config_path, "w", encoding="utf-8") as handle:
                    handle.write(serialized)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return full_config


def append_whitelist_term(config: dict, key: str, term: str) -> tuple[dict, bool]:
    updated = deepcopy(config)
    whitelists = updated.get("whitelists")
    if not isinstance(whitelists, dict):
        whitelists = {}
        updated["whitelists"] = whitelists
    existing = _dedupe_str_list(_as_str_list(whitelists.get(key)))
    items = _dedupe_str_list(existing + [term])
    added = term.strip() in items and term.strip() not in existing
    whitelists[key] = items
    return updated, added


def append_muted_term(config: dict, key: str, term: str) -> tuple[dict, bool]:
    updated = deepcopy(config)
    preferences = get_preferences(config)
    current = preferences["muted"][key]
    items = _dedupe_str_list(current + [term])
    added = term.strip() in items and term.strip() not in current
    preferences["muted"][key] = items
    updated["preferences"] = preferences
    return updated, added


def first_author_name(authors: str | None) -> str:
    if not authors:
        return ""
    return authors.split(",")[0].strip()
