"""Shared request-body validation helpers for the JSON API.

Mutating endpoints accept JSON bodies whose field types are caller/legacy
controlled. Without guards a wrong-typed value (a number where a string is
expected, a dict where a list is expected, a non-hashable where a set lookup
happens) reaches ``.strip()`` / iteration / ``in`` deep in the view and raises an
opaque 500. These helpers fail fast with a clean ``BadRequest`` (400), which the
``api_bp`` error handler renders as JSON.
"""

from __future__ import annotations

from werkzeug.exceptions import BadRequest


def require_str(payload: dict, key: str, *, allow_empty: bool = False) -> str:
    """Return ``payload[key]`` as a stripped string, or raise ``BadRequest``.

    Missing/empty (when ``allow_empty`` is False) yields ``Missing '<key>'`` to
    preserve the existing message; a wrong type yields ``'<key>' must be a string``.
    """
    value = payload.get(key)
    if value is None:
        if allow_empty:
            return ""
        raise BadRequest(f"Missing '{key}'")
    if not isinstance(value, str):
        raise BadRequest(f"'{key}' must be a string")
    value = value.strip()
    if not value and not allow_empty:
        raise BadRequest(f"Missing '{key}'")
    return value


def optional_str(payload: dict, key: str, default: str = "") -> str:
    """Return a stripped string for ``key`` (``default`` if absent/None), or 400."""
    value = payload.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise BadRequest(f"'{key}' must be a string")
    return value.strip()


def require_list(payload: dict, key: str, default: list | None = None) -> list:
    """Return ``payload[key]`` as a list (``default`` if absent), or raise 400."""
    value = payload.get(key, [] if default is None else default)
    if not isinstance(value, list):
        raise BadRequest(f"'{key}' must be a list")
    return value
