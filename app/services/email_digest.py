"""Daily email digest of matched papers via Gmail API (OAuth2).

Security model:
- Uses OAuth2 with the narrowest scope: ``gmail.send`` (send-only, no inbox read).
- Access tokens are short-lived (~60 min) and auto-refreshed from a stored refresh token.
- Credentials file (``credentials.json``) and token file (``token.json``) must NEVER be
  committed to version control — they are listed in ``.gitignore``.
- All user-generated content is HTML-escaped before rendering.

Setup (one-time):
    1. Create a Google Cloud project and enable the Gmail API.
    2. Create OAuth Client ID credentials (Web application type).
    3. Add the app's ``/settings/gmail-callback`` URL as an authorized redirect URI.
    4. Download ``credentials.json`` to the project root.
    5. Click **Authorize Gmail** in the web UI settings page.
    6. A ``token.json`` file is saved with 600 permissions. Guard it like a password.
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import date, timedelta
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Flask

from app.models import Paper

log = logging.getLogger(__name__)

# Only permission needed: send email. Cannot read, list, or delete.
GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CREDENTIALS_PATH = _PROJECT_ROOT / "credentials.json"
DEFAULT_TOKEN_PATH = _PROJECT_ROOT / "token.json"


def check_gmail_auth_status(
    credentials_path: Path | None = None,
    token_path: Path | None = None,
) -> dict:
    """Check Gmail OAuth status without side effects.

    Returns a dict with keys: ``status``, ``message``.
    Possible statuses: ``connected``, ``no_credentials``, ``no_token``,
    ``expired``, ``invalid``.
    """
    if credentials_path is None:
        credentials_path = DEFAULT_CREDENTIALS_PATH
    if token_path is None:
        token_path = DEFAULT_TOKEN_PATH

    if not credentials_path.exists():
        return {
            "status": "no_credentials",
            "message": (
                "credentials.json not found. Download it from the "
                "Google Cloud Console (OAuth 2.0 Client ID → Web application) "
                "and place it in the project root."
            ),
        }

    if not token_path.exists():
        return {
            "status": "no_token",
            "message": (
                "Gmail not authorized yet. Click 'Authorize Gmail' to "
                "start the OAuth flow."
            ),
        }

    try:
        from google.oauth2.credentials import Credentials

        creds = Credentials.from_authorized_user_file(
            str(token_path), scopes=[GMAIL_SEND_SCOPE]
        )
    except Exception:
        return {
            "status": "invalid",
            "message": "token.json is corrupted or unreadable. Re-authorize Gmail.",
        }

    if creds.expired and creds.refresh_token:
        return {
            "status": "expired",
            "message": (
                "Access token expired. It will auto-refresh on next send, "
                "or click 'Re-authorize' to refresh now."
            ),
        }

    if not creds.valid and not creds.refresh_token:
        return {
            "status": "invalid",
            "message": "Token is invalid and cannot be refreshed. Re-authorize Gmail.",
        }

    return {
        "status": "connected",
        "message": "Gmail is connected and ready to send.",
    }


def start_oauth_flow(
    redirect_uri: str,
    credentials_path: Path | None = None,
) -> dict:
    """Build a Google OAuth2 authorization URL for the web redirect flow.

    Returns a dict with ``success``, ``auth_url``, ``state``, and ``message``.
    The caller must store ``state`` in the session and redirect the user to
    ``auth_url``.  After consent Google will redirect back to *redirect_uri*
    with a ``code`` and ``state`` query parameter.
    """
    if credentials_path is None:
        credentials_path = DEFAULT_CREDENTIALS_PATH

    if not credentials_path.exists():
        return {
            "success": False,
            "auth_url": None,
            "state": None,
            "message": (
                "credentials.json not found. Download it from Google Cloud Console first."
            ),
        }

    try:
        from google_auth_oauthlib.flow import Flow
    except ImportError:
        return {
            "success": False,
            "auth_url": None,
            "state": None,
            "message": (
                "google-auth-oauthlib is not installed. "
                "Run: pip install google-auth google-auth-oauthlib google-api-python-client"
            ),
        }

    try:
        flow = Flow.from_client_secrets_file(
            str(credentials_path),
            scopes=[GMAIL_SEND_SCOPE],
            redirect_uri=redirect_uri,
        )
        auth_url, state = flow.authorization_url(
            access_type="offline",
            prompt="consent",
        )
        return {
            "success": True,
            "auth_url": auth_url,
            "state": state,
            "message": "Redirecting to Google for authorization.",
        }
    except Exception as exc:
        return {
            "success": False,
            "auth_url": None,
            "state": None,
            "message": f"Failed to start OAuth flow: {exc}",
        }


def finish_oauth_flow(
    authorization_response_url: str,
    redirect_uri: str,
    credentials_path: Path | None = None,
    token_path: Path | None = None,
) -> dict:
    """Exchange the authorization code for credentials and save the token.

    *authorization_response_url* is the full URL the user was redirected to
    (including the ``code`` and ``state`` query parameters).

    Returns a dict with ``success`` (bool) and ``message``.
    """
    if credentials_path is None:
        credentials_path = DEFAULT_CREDENTIALS_PATH
    if token_path is None:
        token_path = DEFAULT_TOKEN_PATH

    try:
        from google_auth_oauthlib.flow import Flow
    except ImportError:
        return {
            "success": False,
            "message": (
                "google-auth-oauthlib is not installed. "
                "Run: pip install google-auth google-auth-oauthlib google-api-python-client"
            ),
        }

    try:
        flow = Flow.from_client_secrets_file(
            str(credentials_path),
            scopes=[GMAIL_SEND_SCOPE],
            redirect_uri=redirect_uri,
        )
        flow.fetch_token(authorization_response=authorization_response_url)
        creds = flow.credentials
        token_path.write_text(creds.to_json())
        os.chmod(token_path, 0o600)
        return {"success": True, "message": "Gmail authorized successfully."}
    except Exception as exc:
        return {"success": False, "message": f"OAuth flow failed: {exc}"}


def _load_gmail_credentials(
    credentials_path: Path | None = None,
    token_path: Path | None = None,
):
    """Load and refresh OAuth2 credentials for the Gmail API.

    Raises ``FileNotFoundError`` if ``token.json`` is missing (run setup first).
    Raises ``RuntimeError`` if the token cannot be refreshed.
    """
    if credentials_path is None:
        credentials_path = DEFAULT_CREDENTIALS_PATH
    if token_path is None:
        token_path = DEFAULT_TOKEN_PATH

    if not token_path.exists():
        raise FileNotFoundError(
            f"Token file not found: {token_path}\n"
            "Run 'python gmail_auth_setup.py' to complete the one-time OAuth setup."
        )

    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    creds = Credentials.from_authorized_user_file(str(token_path), scopes=[GMAIL_SEND_SCOPE])

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            raise RuntimeError(
                "Failed to refresh Gmail token. The token may have been revoked.\n"
                "Re-run 'python gmail_auth_setup.py' to re-authorize.\n"
                f"Details: {exc}"
            ) from exc
        # Persist the refreshed token.
        token_path.write_text(creds.to_json())
        os.chmod(token_path, 0o600)

    if not creds.valid:
        raise RuntimeError(
            "Gmail credentials are invalid. Re-run 'python gmail_auth_setup.py'."
        )

    return creds


def _build_gmail_service(creds):
    """Build an authorized Gmail API service client."""
    from googleapiclient.discovery import build

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _get_email_config(app: Flask) -> dict:
    """Read non-sensitive email settings from the scraper config."""
    scraper_config = app.config.get("SCRAPER_CONFIG", {})
    email_cfg = scraper_config.get("email", {})
    return {
        "recipient": email_cfg.get("recipient", ""),
        "subject_prefix": email_cfg.get("subject_prefix", "ArXiv Digest"),
    }


def _query_todays_papers(app: Flask, lookback_hours: int = 26) -> list[Paper]:
    """Return papers scraped within the lookback window, ranked by score."""
    from app.services.text import now_utc

    cutoff = now_utc() - timedelta(hours=lookback_hours)
    with app.app_context():
        return (
            Paper.query
            .filter(Paper.scraped_at >= cutoff, Paper.is_hidden.is_(False))
            .order_by(Paper.paper_score.desc(), Paper.id.desc())
            .all()
        )


def _render_paper_html(paper: Paper) -> str:
    """Render a single paper card as HTML with proper escaping."""
    match_badges = "".join(
        f'<span style="display:inline-block;background:#e0e7ff;color:#3730a3;'
        f'padding:2px 8px;border-radius:12px;font-size:12px;margin-right:4px;">'
        f'{escape(t.strip())}</span>'
        for t in (paper.match_type or "").split("+")
        if t.strip()
    )

    topic_tags = ""
    if paper.topic_tags:
        tags = paper.topic_tags_list
        topic_tags = " ".join(
            f'<span style="display:inline-block;background:#f0fdf4;color:#166534;'
            f'padding:2px 6px;border-radius:8px;font-size:11px;margin-right:3px;">'
            f'{escape(t)}</span>'
            for t in tags[:6]
        )

    resource_links_html = ""
    for res in paper.resource_links_list:
        url = escape(res.get("url", ""), quote=True)
        label = escape(res.get("type", "link"))
        resource_links_html += (
            f' <a href="{url}" style="color:#2563eb;font-size:12px;'
            f'margin-right:6px;">[{label}]</a>'
        )

    return f"""
    <div style="border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin-bottom:12px;">
      <div style="margin-bottom:4px;">{match_badges}</div>
      <a href="{escape(paper.link, quote=True)}"
         style="color:#1d4ed8;font-size:16px;font-weight:600;text-decoration:none;">
        {escape(paper.title)}
      </a>
      <div style="color:#6b7280;font-size:13px;margin:4px 0;">
        {escape(paper.authors[:200])}
      </div>
      <div style="color:#374151;font-size:13px;margin:6px 0;">
        {escape(paper.summary_text or paper.abstract_text[:300])}
      </div>
      <div style="margin-top:6px;">
        {topic_tags}
        {resource_links_html}
        <span style="float:right;color:#9ca3af;font-size:12px;">
          Score: {paper.paper_score:.1f}
        </span>
      </div>
    </div>
    """


def _build_email_body(papers: list[Paper], today: date) -> str:
    """Compose the full HTML email body."""
    if not papers:
        paper_cards = (
            '<p style="color:#6b7280;text-align:center;padding:40px 0;">'
            "No new matching papers found today.</p>"
        )
    else:
        paper_cards = "\n".join(_render_paper_html(p) for p in papers)

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#f9fafb;font-family:
  -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="max-width:680px;margin:0 auto;padding:24px;">
    <div style="background:white;border-radius:12px;padding:24px;
                box-shadow:0 1px 3px rgba(0,0,0,0.1);">
      <h1 style="font-size:22px;color:#111827;margin:0 0 4px;">
        ArXiv CV Digest
      </h1>
      <p style="color:#6b7280;font-size:14px;margin:0 0 20px;">
        {escape(today.strftime('%A, %B %d, %Y'))} &mdash;
        {len(papers)} paper{"s" if len(papers) != 1 else ""} matched
      </p>
      {paper_cards}
      <hr style="border:none;border-top:1px solid #e5e7eb;margin:20px 0 12px;">
      <p style="color:#9ca3af;font-size:11px;text-align:center;margin:0;">
        Sent by ArXiv CV Scraper &middot; Manage your whitelists in the dashboard
      </p>
    </div>
  </div>
</body>
</html>"""


def send_digest(app: Flask, *, dry_run: bool = False) -> dict:
    """Query today's papers and send a digest via Gmail API.

    Returns a dict with keys: ``papers_count``, ``sent``, ``recipient``.
    """
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    creds = _load_gmail_credentials()
    email_cfg = _get_email_config(app)

    recipient = email_cfg["recipient"]
    if not recipient:
        raise ValueError(
            "No recipient configured. Set 'email.recipient' in config.yaml."
        )

    subject_prefix = email_cfg["subject_prefix"]
    papers = _query_todays_papers(app)
    today = date.today()

    subject = f"{subject_prefix} — {today.strftime('%b %d, %Y')} ({len(papers)} papers)"
    html_body = _build_email_body(papers, today)

    msg = MIMEMultipart("alternative")
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    if dry_run:
        log.info("Dry run — email not sent (would send to %s)", recipient)
        return {"papers_count": len(papers), "sent": False, "recipient": recipient}

    service = _build_gmail_service(creds)
    raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
    service.users().messages().send(
        userId="me",
        body={"raw": raw_message},
    ).execute()

    log.info("Digest sent to %s (%d papers)", recipient, len(papers))
    return {"papers_count": len(papers), "sent": True, "recipient": recipient}
