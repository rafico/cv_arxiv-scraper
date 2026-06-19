"""Classic / modern UI selection.

The redesigned ("modern") UI is the default. Users can opt into the
pre-redesign ("classic") UI — vendored under ``templates/classic/`` and styled
by ``static/style.classic.css`` — via the ``ui_mode`` cookie. The cookie is set
by the ``ui`` blueprint (see :mod:`app.routes.ui`).
"""

from __future__ import annotations

from flask import render_template, request

UI_COOKIE = "ui_mode"
CLASSIC = "classic"


def ui_is_classic() -> bool:
    """True when the current request opted into the classic UI."""
    return request.cookies.get(UI_COOKIE) == CLASSIC


def render_ui(template_name: str, **context: object) -> str:
    """Render ``template_name``, preferring its ``classic/`` variant when toggled.

    Classic templates live alongside the modern ones under ``classic/`` and use
    distinct names so Jinja's compiled-template cache never collides between the
    two themes.
    """
    if ui_is_classic():
        return render_template(f"classic/{template_name}", **context)
    return render_template(template_name, **context)
