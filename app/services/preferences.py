"""Config-backed product preferences and lightweight recommendations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

DEFAULT_PREFERENCES = {
    "ranking": {
        "author_weight": 44.0,
        "affiliation_weight": 26.0,
        "title_weight": 14.0,
        "ai_weight": 5.0,
        "citation_weight": 0.05,
        "freshness_half_life_days": 14.0,
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

    muted = raw.get("muted", {})
    if isinstance(muted, dict):
        for key in merged["muted"]:
            merged["muted"][key] = _dedupe_str_list(list(muted.get(key, [])))

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

    muted = preferences["muted"]
    for key in muted:
        raw = form.get(f"muted_{key}", "")
        muted[key] = _dedupe_str_list(raw.splitlines())

    updated["preferences"] = preferences
    return updated


def save_config(config_path: Path, full_config: dict) -> dict:
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(full_config, handle, default_flow_style=False, sort_keys=False)
    return full_config


def append_whitelist_term(config: dict, key: str, term: str) -> tuple[dict, bool]:
    updated = deepcopy(config)
    whitelists = updated.setdefault("whitelists", {})
    items = _dedupe_str_list(list(whitelists.get(key, [])) + [term])
    added = term.strip() in items and term.strip() not in config.get("whitelists", {}).get(key, [])
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
