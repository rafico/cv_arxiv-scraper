"""One-tap 👍/👎 feedback links from digest emails.

GET-with-side-effect is intentional here: email clients cannot POST, so the
digest buttons are plain links. Authentication/anti-CSRF is the signed,
expiring itsdangerous token minted by the digest builder (bound to this app's
SECRET_KEY) — that is why, unlike the other mutating routes, this endpoint
deliberately does NOT call validate_csrf_token().
"""

from __future__ import annotations

from html import escape

from flask import current_app, request

from app.routes.api import api_bp

_ACTION_LABELS = {"save": "Saved", "skip": "Skipped"}


def _one_tap_page(heading: str, message: str, *, ok: bool = True) -> str:
    """Tiny self-contained confirmation page (no template inheritance needed)."""
    color = "#166534" if ok else "#991b1b"
    icon = "✓" if ok else "✕"
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(heading)}</title></head>
<body style="margin:0;padding:0;background:#f9fafb;font-family:
  -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="max-width:420px;margin:15vh auto 0;padding:24px;text-align:center;">
    <div style="background:white;border-radius:12px;padding:32px 24px;box-shadow:0 1px 3px rgba(0,0,0,0.1);">
      <div style="font-size:36px;color:{color};margin-bottom:8px;">{icon}</div>
      <h1 style="font-size:18px;color:#111827;margin:0 0 8px;">{escape(heading)}</h1>
      <p style="color:#6b7280;font-size:14px;margin:0;">{escape(message)}</p>
    </div>
  </div>
</body>
</html>"""


@api_bp.route("/feedback/one-tap", methods=["GET"])
def one_tap_feedback():
    from itsdangerous import BadSignature, SignatureExpired

    from app.services.email_digest import load_one_tap_token

    token = request.args.get("token", "")
    if not token:
        return _one_tap_page("Missing token", "This link is missing its token.", ok=False), 400

    try:
        data = load_one_tap_token(current_app._get_current_object(), token)
    except SignatureExpired:
        return (
            _one_tap_page("Link expired", "Digest links are valid for 7 days. Open the dashboard instead.", ok=False),
            400,
        )
    except BadSignature:
        return _one_tap_page("Invalid link", "This link could not be verified.", ok=False), 400

    paper_id = data["p"]
    action = data["a"]
    label = _ACTION_LABELS[action]

    # Attribute the feedback to the profile whose digest section carried this
    # paper. Tokens without "pr" (pre-Wave-3, or a non-profile section) default
    # to the default profile.
    profile_id = data.get("pr")
    if profile_id is None:
        try:
            from app.services.profiles import get_default_profile

            profile_id = int(get_default_profile().id)
        except Exception:
            profile_id = None

    from app.models import PaperFeedback
    from app.services import apply_feedback_action

    # apply_feedback_action is a toggle; a re-clicked email link must be
    # idempotent, so an already-recorded action short-circuits to a confirmation.
    existing = PaperFeedback.query.filter_by(paper_id=paper_id, action=action).first()
    if existing is not None:
        return _one_tap_page(f"{label} ✓", "Already recorded — you can close this tab.")

    try:
        apply_feedback_action(paper_id, action, profile_id=profile_id)
    except LookupError:
        return _one_tap_page("Paper not found", "This paper no longer exists in your library.", ok=False), 404

    return _one_tap_page(f"{label} ✓", "You can close this tab.")
