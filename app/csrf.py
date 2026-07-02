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


def rotate_csrf_token() -> str:
    """Replace the session's CSRF token with a fresh one and return it.

    Called by settings routes after a successful mutation so a leaked token
    cannot be replayed.
    """
    csrf_token = token_urlsafe(32)
    session[CSRF_SESSION_KEY] = csrf_token
    return csrf_token


def validate_csrf_token(submitted_token: str | None = None) -> None:
    token = submitted_token or request.headers.get("X-CSRF-Token", "") or request.form.get("csrf_token", "")
    expected_token = session.get(CSRF_SESSION_KEY, "")
    # secrets.compare_digest raises TypeError on a non-ASCII str; a real token is
    # url-safe ASCII, so treat any non-ASCII submission as invalid (400) rather than
    # letting the TypeError become an unhandled 500 on routes without an error handler.
    if not token or not expected_token or not token.isascii() or not compare_digest(token, expected_token):
        abort(400, description="Invalid CSRF token")
