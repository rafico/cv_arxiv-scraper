"""Shared CSRF helpers for browser-driven routes."""

from __future__ import annotations

from secrets import compare_digest, token_urlsafe

from flask import abort, request, session

CSRF_SESSION_KEY = "settings_csrf_token"


def get_or_create_csrf_token() -> str:
    csrf_token = session.get(CSRF_SESSION_KEY)
    if csrf_token:
        return csrf_token

    csrf_token = token_urlsafe(32)
    session[CSRF_SESSION_KEY] = csrf_token
    return csrf_token


def validate_csrf_token(submitted_token: str | None = None) -> None:
    token = (
        submitted_token
        or request.headers.get("X-CSRF-Token", "")
        or request.form.get("csrf_token", "")
        or request.args.get("csrf_token", "")
    )
    expected_token = session.get(CSRF_SESSION_KEY, "")
    if not token or not expected_token or not compare_digest(token, expected_token):
        abort(400, description="Invalid CSRF token")
