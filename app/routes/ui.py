"""Endpoint that switches between the modern and classic UIs via a cookie."""

from __future__ import annotations

from urllib.parse import urlparse

from flask import Blueprint, redirect, request, url_for

from app.ui import CLASSIC, UI_COOKIE

ui_bp = Blueprint("ui", __name__)

_ONE_YEAR = 60 * 60 * 24 * 365


def _safe_redirect_target() -> str:
    """Return the referrer if it's a same-host relative target, else the inbox."""
    referrer = request.referrer
    if referrer:
        parsed = urlparse(referrer)
        if not parsed.netloc or parsed.netloc == request.host:
            return referrer
    return url_for("dashboard.index")


@ui_bp.route("/ui/<mode>")
def set_mode(mode: str):
    """Persist the UI choice and bounce back to the page the user came from.

    The modern and classic UIs share identical URLs, so redirecting to the
    referrer re-renders the same page in the newly selected theme.
    """
    response = redirect(_safe_redirect_target())
    if mode == CLASSIC:
        response.set_cookie(UI_COOKIE, CLASSIC, max_age=_ONE_YEAR, samesite="Lax")
    else:
        response.delete_cookie(UI_COOKIE)
    return response
