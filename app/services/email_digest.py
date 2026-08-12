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
import math
import os
import re
from datetime import date, timedelta
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Flask

from app.models import DigestRun, Paper, db
from app.services.ranking import combined_rank_score, rank_score_order_expr
from app.services.secret_files import write_secret_file
from app.services.text import now_utc, utc_today

LOGGER = logging.getLogger(__name__)

# Only permission needed: send email. Cannot read, list, or delete.
GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CREDENTIALS_PATH = _PROJECT_ROOT / "credentials.json"
DEFAULT_TOKEN_PATH = _PROJECT_ROOT / "token.json"

# ── Digest 2.0 knobs (config `digest:` block, Settings → Automation → Digest) ──
DIGEST_WEEKDAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
DEFAULT_DIGEST_MAX_PAPERS = 15
DEFAULT_LOOKBACK_HOURS = 26
# Widen the window ("catch-up digest") when the last successful send is older than this.
CATCH_UP_AFTER_HOURS = 48
CATCH_UP_MAX_DAYS = 7
# One-tap 👍/👎 links stay valid this long (signed, itsdangerous).
ONE_TAP_MAX_AGE_SECONDS = 7 * 24 * 3600
_ONE_TAP_SALT = "digest-one-tap"
# Total bytes of inline (CID) figure images attached per email; figures past the
# cap are silently dropped so the message stays small on mobile connections.
MAX_INLINE_FIGURE_BYTES = 1_500_000
INLINE_FIGURE_PAPERS = 5
# Mirrors dashboard._thumbnail_storage_key so digest figures resolve the same files.
_STORAGE_KEY_RE = re.compile(r"^[A-Za-z0-9._\-]+(?:/[A-Za-z0-9._\-]+)?$")


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
                "Upload credentials.json to get started. Create an OAuth 2.0 "
                "Client ID (Web application type) in Google Cloud Console, "
                "then upload the JSON file below."
            ),
            "action": "upload_credentials",
            "help_url": "https://console.cloud.google.com/apis/credentials",
        }

    if not token_path.exists():
        return {
            "status": "no_token",
            "message": ("Credentials uploaded. Click 'Authorize Gmail' to connect your account (send-only access)."),
            "action": "authorize",
        }

    try:
        from google.oauth2.credentials import Credentials

        creds = Credentials.from_authorized_user_file(str(token_path), scopes=[GMAIL_SEND_SCOPE])
    except Exception:
        return {
            "status": "invalid",
            "message": ("token.json is corrupted or unreadable. Click 'Re-authorize' to generate a fresh token."),
            "action": "reauthorize",
        }

    if creds.expired and creds.refresh_token:
        return {
            "status": "expired",
            "message": (
                "Access token expired but will auto-refresh on next send. "
                "You can also click 'Re-authorize' to refresh now."
            ),
            "action": "reauthorize",
        }

    if not creds.valid and not creds.refresh_token:
        return {
            "status": "invalid",
            "message": (
                "Token is invalid and has no refresh token. This usually "
                "means the token was revoked. Click 'Re-authorize' to fix."
            ),
            "action": "reauthorize",
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
            "message": ("credentials.json not found. Download it from Google Cloud Console first."),
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
        write_secret_file(token_path, creds.to_json())
        return {"success": True, "message": "Gmail authorized successfully."}
    except Exception as exc:
        msg = str(exc)
        if "redirect_uri_mismatch" in msg.lower() or "redirect uri" in msg.lower():
            return {
                "success": False,
                "message": (
                    "Redirect URI mismatch. The callback URL configured in "
                    "Google Cloud Console doesn't match this app's URL. "
                    "Check that the authorized redirect URI matches exactly."
                ),
            }
        return {"success": False, "message": f"OAuth token exchange failed: {exc}"}


def get_setup_instructions(
    credentials_path: Path | None = None,
    token_path: Path | None = None,
    callback_uri: str = "",
    recipient: str = "",
) -> list[dict]:
    """Return step-by-step setup checklist for the settings template.

    Each step has keys: ``step``, ``label``, ``complete``, ``description``.
    """
    if credentials_path is None:
        credentials_path = DEFAULT_CREDENTIALS_PATH
    if token_path is None:
        token_path = DEFAULT_TOKEN_PATH

    has_creds = credentials_path.exists()
    has_token = token_path.exists()
    has_recipient = bool(recipient)

    token_valid = False
    if has_token:
        try:
            from google.oauth2.credentials import Credentials

            creds = Credentials.from_authorized_user_file(str(token_path), scopes=[GMAIL_SEND_SCOPE])
            token_valid = creds.valid or (creds.expired and creds.refresh_token)
        except Exception:
            pass

    return [
        {
            "step": 1,
            "label": "Create Google Cloud credentials",
            "complete": has_creds,
            "description": (
                "Create an OAuth 2.0 Client ID (Web application) in Google Cloud Console and upload credentials.json."
            ),
        },
        {
            "step": 2,
            "label": "Set redirect URI",
            "complete": has_creds,
            "description": (
                f"Add this as an authorized redirect URI in Google Cloud Console: {callback_uri}"
                if callback_uri
                else "Add the app's callback URL as an authorized redirect URI."
            ),
        },
        {
            "step": 3,
            "label": "Authorize Gmail",
            "complete": token_valid,
            "description": ("Click 'Authorize Gmail' to grant send-only access."),
        },
        {
            "step": 4,
            "label": "Set recipient email",
            "complete": has_recipient,
            "description": "Enter the email address where digests should go.",
        },
    ]


def validate_credentials_redirect_uris(
    callback_uri: str,
    credentials_path: Path | None = None,
) -> dict:
    """Check if the app's callback URI is in credentials.json redirect_uris.

    Returns dict with ``match`` (bool) and ``message``.
    """
    import json as _json

    if credentials_path is None:
        credentials_path = DEFAULT_CREDENTIALS_PATH

    if not credentials_path.exists():
        return {"match": False, "message": "No credentials.json found."}

    try:
        data = _json.loads(credentials_path.read_text(encoding="utf-8"))
    except Exception:
        return {"match": False, "message": "Could not read credentials.json."}

    web = data.get("web", {})
    redirect_uris = web.get("redirect_uris", [])

    if not redirect_uris:
        return {
            "match": False,
            "message": (
                "No redirect URIs found in credentials.json. This is normal "
                "for newly created credentials -- add the callback URI in "
                "Google Cloud Console."
            ),
        }

    if callback_uri in redirect_uris:
        return {"match": True, "message": "Callback URI matches credentials.json."}

    return {
        "match": False,
        "message": (
            f"Callback URI mismatch. This app uses '{callback_uri}' but "
            f"credentials.json has: {', '.join(redirect_uris)}. "
            f"Update the authorized redirect URIs in Google Cloud Console."
        ),
    }


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
        write_secret_file(token_path, creds.to_json())

    if not creds.valid:
        raise RuntimeError("Gmail credentials are invalid. Re-run 'python gmail_auth_setup.py'.")

    return creds


def _build_gmail_service(creds):
    """Build an authorized Gmail API service client."""
    from googleapiclient.discovery import build

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _get_email_config(app: Flask) -> dict:
    """Read non-sensitive email settings from the scraper config."""
    scraper_config = app.config.get("SCRAPER_CONFIG", {})
    email_cfg = scraper_config.get("email")
    if not isinstance(email_cfg, dict):
        email_cfg = {}
    return {
        "recipient": email_cfg.get("recipient", ""),
        "subject_prefix": email_cfg.get("subject_prefix", "ArXiv Digest"),
    }


def get_digest_config(app: Flask) -> dict:
    """Read the ``digest:`` config block with defensive defaults.

    A hand-edited block must never crash a send or a settings render, so every
    field falls back to its default on any type problem.
    """
    scraper_config = app.config.get("SCRAPER_CONFIG", {})
    raw = scraper_config.get("digest") if isinstance(scraper_config, dict) else None
    if not isinstance(raw, dict):
        raw = {}

    weekdays: list[str] = []
    raw_weekdays = raw.get("weekdays")
    if isinstance(raw_weekdays, list):
        normalized = {str(day).strip().lower()[:3] for day in raw_weekdays}
        weekdays = [key for key in DIGEST_WEEKDAY_KEYS if key in normalized]
    if not weekdays:
        weekdays = list(DIGEST_WEEKDAY_KEYS)

    try:
        min_score = float(raw.get("min_score", 0.0))
        if not math.isfinite(min_score):
            min_score = 0.0
    except (TypeError, ValueError):
        min_score = 0.0

    try:
        max_papers = int(raw.get("max_papers", DEFAULT_DIGEST_MAX_PAPERS))
    except (TypeError, ValueError):
        max_papers = DEFAULT_DIGEST_MAX_PAPERS
    max_papers = max(1, min(100, max_papers))

    try:
        exploration_slots = int(raw.get("exploration_slots", 2))
    except (TypeError, ValueError):
        exploration_slots = 2
    exploration_slots = max(0, min(10, exploration_slots))

    # Weekday for the "what happened in your field" synthesis brief; off unless
    # a valid day is configured.
    synthesis_weekday = str(raw.get("synthesis_weekday", "") or "").strip().lower()[:3]
    if synthesis_weekday not in DIGEST_WEEKDAY_KEYS:
        synthesis_weekday = None

    base_url = raw.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        # Mirrors run.py's default bind (127.0.0.1, PORT env or 5000). The app has
        # no canonical external URL — digests may be built outside a request.
        base_url = f"http://127.0.0.1:{os.environ.get('PORT', '5000')}"
    return {
        "weekdays": weekdays,
        "min_score": min_score,
        "max_papers": max_papers,
        "exploration_slots": exploration_slots,
        "synthesis_weekday": synthesis_weekday,
        "base_url": base_url.strip().rstrip("/"),
    }


def weekday_allowed(digest_cfg: dict, today: date | None = None) -> bool:
    """True when the digest is scheduled to go out on ``today`` (UTC)."""
    key = DIGEST_WEEKDAY_KEYS[(today or utc_today()).weekday()]
    return key in digest_cfg.get("weekdays", DIGEST_WEEKDAY_KEYS)


# ── One-tap feedback tokens ──────────────────────────────────────────────


def _one_tap_serializer(secret_key: str):
    from itsdangerous import URLSafeTimedSerializer

    return URLSafeTimedSerializer(secret_key, salt=_ONE_TAP_SALT)


def make_one_tap_token(
    app: Flask,
    paper_id: int,
    action: str,
    digest_run_id: int | None = None,
    profile_id: int | None = None,
) -> str:
    """Sign a (paper_id, action, digest_run_id, profile_id) one-tap payload.

    ``profile_id`` attributes the feedback to the profile whose digest section
    the paper appeared in. Omitted for pre-Wave-3 tokens (defaults to the
    default profile on load).
    """
    payload = {"p": int(paper_id), "a": str(action), "r": digest_run_id}
    if profile_id is not None:
        payload["pr"] = int(profile_id)
    return _one_tap_serializer(app.config["SECRET_KEY"]).dumps(payload)


def load_one_tap_token(app: Flask, token: str, max_age: int = ONE_TAP_MAX_AGE_SECONDS) -> dict:
    """Verify and decode a one-tap token.

    Raises ``itsdangerous.SignatureExpired`` past ``max_age`` and
    ``itsdangerous.BadSignature`` on tampering or a malformed payload. Back-compat:
    a token without ``pr`` (pre-Wave-3, or a non-profile section) decodes fine and
    the caller defaults it to the default profile.
    """
    from itsdangerous import BadSignature

    data = _one_tap_serializer(app.config["SECRET_KEY"]).loads(token, max_age=max_age)
    if not isinstance(data, dict) or not isinstance(data.get("p"), int) or data.get("a") not in ("save", "skip"):
        raise BadSignature("Malformed one-tap payload")
    if "pr" in data and not isinstance(data["pr"], int):
        raise BadSignature("Malformed one-tap payload")
    return data


# ── Catch-up window / figures / saved-search alerts ─────────────────────


def _resolve_lookback(app: Flask) -> dict:
    """Pick the query window: 26h normally, widened after a send gap (cap 7 days)."""
    with app.app_context():
        last = DigestRun.query.filter(DigestRun.status == "success").order_by(DigestRun.started_at.desc()).first()
        last_run_at = last.started_at if last is not None else None

    state = {
        "catch_up": False,
        "lookback_hours": DEFAULT_LOOKBACK_HOURS,
        "days": None,
        "last_run_at": last_run_at,
    }
    if last_run_at is None:
        return state

    gap_hours = (now_utc() - last_run_at).total_seconds() / 3600.0
    if gap_hours <= CATCH_UP_AFTER_HOURS:
        return state

    # +2h of overlap so a paper scraped just before the last send isn't dropped.
    lookback_hours = int(math.ceil(min(gap_hours + 2.0, CATCH_UP_MAX_DAYS * 24.0)))
    state.update(
        {
            "catch_up": True,
            "lookback_hours": lookback_hours,
            "days": max(2, min(CATCH_UP_MAX_DAYS, math.ceil(lookback_hours / 24.0))),
        }
    )
    return state


def _figure_storage_key(paper: Paper) -> str | None:
    candidate: str | None = None
    if paper.arxiv_id:
        candidate = paper.arxiv_id
    elif paper.link:
        candidate = paper.link.rstrip("/").split("/")[-1]
    if candidate and _STORAGE_KEY_RE.fullmatch(candidate):
        return candidate
    return None


def _first_figure_path(app: Flask, paper: Paper) -> Path | None:
    from app.services.thumbnail_generator import figure_paths_for

    storage_key = _figure_storage_key(paper)
    if not storage_key or not app.static_folder:
        return None
    try:
        paths = figure_paths_for(storage_key, app.static_folder)
    except Exception:  # figures are decorative; never break digest assembly
        LOGGER.debug("Figure lookup failed for %s", storage_key, exc_info=True)
        return None
    return paths[0] if paths else None


def _preview_figure_srcs(app: Flask, papers: list[Paper], base_url: str) -> dict[int, str]:
    """HTTP figure URLs for the browser preview (email uses CID parts instead)."""
    srcs: dict[int, str] = {}
    for paper in papers[:INLINE_FIGURE_PAPERS]:
        if _first_figure_path(app, paper) is not None:
            srcs[paper.id] = f"{base_url}/papers/{paper.id}/figures/1.png"
    return srcs


def _collect_figure_attachments(app: Flask, papers: list[Paper]) -> tuple[list[tuple[str, bytes]], dict[int, str]]:
    """Read figure files for the top papers as (cid, bytes) parts, capped in size."""
    attachments: list[tuple[str, bytes]] = []
    srcs: dict[int, str] = {}
    total = 0
    for paper in papers[:INLINE_FIGURE_PAPERS]:
        path = _first_figure_path(app, paper)
        if path is None:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if not data or total + len(data) > MAX_INLINE_FIGURE_BYTES:
            continue
        total += len(data)
        cid = f"fig-{paper.id}"
        attachments.append((cid, data))
        srcs[paper.id] = f"cid:{cid}"
    return attachments, srcs


def _collect_saved_search_alerts(app: Flask, *, since, exclude_paper_ids: list[int]) -> list[dict]:
    """Saved searches flagged notify_on_match → new matches since the last digest."""
    from app.services.saved_search import run_notify_searches

    try:
        with app.app_context():
            entries = run_notify_searches(since=since, exclude_paper_ids=exclude_paper_ids)
            return [{"name": entry["search"].name, "papers": entry["papers"]} for entry in entries]
    except Exception:  # alerts are additive; a broken saved search must not kill the digest
        LOGGER.warning("Saved-search alert assembly failed", exc_info=True)
        return []


def _create_digest_run(
    app: Flask,
    *,
    recipient: str,
    subject: str,
    papers_count: int,
    preview_only: bool,
) -> int:
    with app.app_context():
        run = DigestRun(
            status="running",
            recipient=recipient,
            subject=subject,
            papers_count=papers_count,
            preview_only=preview_only,
            started_at=now_utc(),
        )
        db.session.add(run)
        db.session.commit()
        return int(run.id)


def _finish_digest_run(app: Flask, digest_run_id: int | None, *, status: str, error_message: str | None = None) -> None:
    if digest_run_id is None:
        return

    with app.app_context():
        run = db.session.get(DigestRun, digest_run_id)
        if run is None:
            return
        run.status = status
        run.error_message = error_message
        run.finished_at = now_utc()
        db.session.commit()


def get_digest_history(limit: int = 6) -> list[DigestRun]:
    return DigestRun.query.order_by(DigestRun.started_at.desc()).limit(limit).all()


def _digest_profiles(app: Flask) -> list:
    """Interest profiles flagged for the digest (default first). Best-effort."""
    try:
        from app.models import InterestProfile

        with app.app_context():
            from app.services.profiles import ensure_default_profile

            ensure_default_profile()
            return (
                InterestProfile.query.filter(InterestProfile.include_in_digest.is_(True))
                .order_by(InterestProfile.is_default.desc(), InterestProfile.created_at.asc())
                .all()
            )
    except Exception:  # pragma: no cover - digest must survive a profile-table problem
        LOGGER.warning("Digest profile lookup failed (non-fatal)", exc_info=True)
        return []


def _rank_papers_for_profile(app: Flask, papers: list[Paper], profile) -> list[Paper]:
    """Reorder ``papers`` by ``profile``'s learned model + NL description.

    Best-effort: with no model/embeddings the input order (stored rank score) is
    kept, so a digest section never breaks because a profile has no model yet.
    """
    if not papers:
        return papers
    try:
        from app.services import learned_ranker
        from app.services.embeddings import get_embedding_service
        from app.services.profiles import profile_ref

        ref = profile_ref(profile)
        model = learned_ranker.ensure_learned_model(app, profile=ref)
        description_vector = learned_ranker._description_vector(ref)
        if model is None and description_vector is None:
            return papers
        blend = 0.7
        try:
            with app.app_context():
                blend = float(learned_ranker.learned_preferences(app.config.get("SCRAPER_CONFIG")).get("blend", 0.7))
        except Exception:  # pragma: no cover - best-effort
            pass
        service = get_embedding_service(app)
        found, vectors = service.get_paper_vectors([p.id for p in papers])
        score_by_id: dict[int, float] = {}
        for idx, pid in enumerate(found):
            signal, _source = learned_ranker.interest_signal(vectors[idx], None, model, blend, description_vector)
            if signal is not None:
                score_by_id[pid] = float(signal)
        return sorted(papers, key=lambda p: score_by_id.get(p.id, -2.0), reverse=True)
    except Exception:  # pragma: no cover - ranking is best-effort
        LOGGER.warning("Per-profile digest ranking failed (non-fatal)", exc_info=True)
        return papers


def build_digest_preview(app: Flask) -> dict:
    email_cfg = _get_email_config(app)
    digest_cfg = get_digest_config(app)
    window = _resolve_lookback(app)
    max_papers = digest_cfg["max_papers"]
    profiles = _digest_profiles(app)
    default_profile_id = next((int(p.id) for p in profiles if p.is_default), None)

    sections: list[dict] | None = None
    if len(profiles) > 1:
        # One section per profile, each re-ranked by that profile's model.
        pool = _query_todays_papers(
            app,
            lookback_hours=window["lookback_hours"],
            min_score=digest_cfg["min_score"],
            max_papers=max(max_papers * 4, max_papers),
        )
        sections = []
        seen: set[int] = set()
        papers = []
        for profile in profiles:
            ranked = _rank_papers_for_profile(app, list(pool), profile)[:max_papers]
            if not ranked:
                continue
            sections.append({"name": profile.name, "profile_id": int(profile.id), "papers": ranked})
            for p in ranked:
                if p.id not in seen:
                    seen.add(p.id)
                    papers.append(p)
        if not sections:
            sections = None
    if sections is None:
        papers = _query_todays_papers(
            app,
            lookback_hours=window["lookback_hours"],
            min_score=digest_cfg["min_score"],
            max_papers=max_papers,
        )

    alerts_since = window["last_run_at"] or (now_utc() - timedelta(hours=window["lookback_hours"]))
    alerts = _collect_saved_search_alerts(app, since=alerts_since, exclude_paper_ids=[p.id for p in papers])

    # Exploration draws from the leftovers only: never the main list, never a
    # saved-search alert (an explicit notify beats a random exploration slot).
    exploration: list[Paper] = []
    if digest_cfg["exploration_slots"] and papers:
        alert_ids = [p.id for alert in alerts for p in alert["papers"]]
        exploration = _query_exploration_papers(
            app,
            window["lookback_hours"],
            exclude_ids=[p.id for p in papers] + alert_ids,
            count=digest_cfg["exploration_slots"],
        )
    today = utc_today()

    # Weekly field-synthesis brief on the configured weekday only (LLM optional;
    # degrades to topic labels + counts without one).
    synthesis = None
    if digest_cfg["synthesis_weekday"] == DIGEST_WEEKDAY_KEYS[today.weekday()]:
        try:
            from app.services.corpus_analysis import synthesize_recent_topics
            from app.services.rag import build_llm_client

            synthesis = synthesize_recent_topics(llm_client=build_llm_client(app))
            if not synthesis.get("topics"):
                synthesis = None
        except Exception:  # noqa: BLE001 — the digest must send without the brief
            LOGGER.warning("Field synthesis unavailable for this digest", exc_info=True)

    catch_up_label = None
    if window["catch_up"]:
        catch_up_label = f"Catch-up digest — last {window['days']} days"
        subject = f"{email_cfg['subject_prefix']} — {catch_up_label} ({len(papers)} papers)"
    else:
        subject = f"{email_cfg['subject_prefix']} — {today.strftime('%b %d, %Y')} ({len(papers)} papers)"

    ctx = {
        "base_url": digest_cfg["base_url"],
        "token_for": lambda paper_id, action, profile_id=None: make_one_tap_token(
            app, paper_id, action, None, profile_id
        ),
        "figure_srcs": _preview_figure_srcs(app, list(papers) + list(exploration), digest_cfg["base_url"]),
        "catch_up_label": catch_up_label,
        "alerts": alerts,
        "sections": sections,
        "default_profile_id": default_profile_id,
        "exploration": exploration,
        "synthesis": synthesis,
    }
    return {
        "recipient": email_cfg["recipient"],
        "subject": subject,
        "papers_count": len(papers),
        "papers": papers,
        "html": _build_email_body(papers, today, ctx),
        "catch_up": window["catch_up"],
        "catch_up_label": catch_up_label,
        "lookback_hours": window["lookback_hours"],
        "alerts": alerts,
        "sections": sections,
        "default_profile_id": default_profile_id,
        "exploration": exploration,
        "synthesis": synthesis,
    }


def get_digest_status_snapshot(app: Flask) -> dict:
    email_cfg = _get_email_config(app)
    latest = DigestRun.query.order_by(DigestRun.started_at.desc()).first()
    preview = build_digest_preview(app)
    return {
        "recipient": email_cfg["recipient"],
        "papers_ready": preview["papers_count"],
        "preview_subject": preview["subject"],
        "latest_run": latest,
    }


def _query_exploration_papers(app: Flask, lookback_hours: int, exclude_ids: list[int], count: int) -> list[Paper]:
    """Sample papers from *outside* the user's usual lane for labeled exploration.

    Scholar Inbox's recipe: a couple of clearly-labeled slots per digest whose
    ratings teach the model where the user's interests end. Prefers papers the
    interest model knows least about (NULL similarity first, then lowest), with
    a random tiebreak so the slots vary between digests.
    """
    from sqlalchemy import func

    cutoff = now_utc() - timedelta(hours=lookback_hours)
    with app.app_context():
        query = Paper.query.filter(Paper.scraped_at >= cutoff, Paper.is_hidden.is_(False))
        if exclude_ids:
            query = query.filter(Paper.id.notin_(exclude_ids))
        return (
            query.order_by(
                func.coalesce(Paper.interest_similarity, -1.0).asc(),
                func.random(),
            )
            .limit(count)
            .all()
        )


def _query_todays_papers(
    app: Flask,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    *,
    min_score: float = 0.0,
    max_papers: int | None = None,
) -> list[Paper]:
    """Return papers scraped within the lookback window, ranked by score."""
    cutoff = now_utc() - timedelta(hours=lookback_hours)
    with app.app_context():
        query = Paper.query.filter(Paper.scraped_at >= cutoff, Paper.is_hidden.is_(False))
        if min_score > 0:
            query = query.filter(rank_score_order_expr() >= min_score)
        query = query.order_by(
            rank_score_order_expr().desc(),
            Paper.publication_dt.desc(),
            Paper.scraped_at.desc(),
        )
        if max_papers is not None and max_papers > 0:
            query = query.limit(max_papers)
        return query.all()


def _score_chip(paper: Paper) -> str:
    score = combined_rank_score(float(paper.paper_score or 0.0), int(paper.feedback_score or 0))
    return (
        '<span style="display:inline-block;background:#eef2ff;color:#3730a3;padding:2px 10px;'
        f'border-radius:999px;font-size:12px;font-weight:600;">Score: {score:.1f}</span>'
    )


def _one_tap_buttons(paper: Paper, ctx: dict | None, profile_id: int | None = None) -> str:
    """👍 Save / 👎 Skip links that hit the local one-tap API (token-signed GETs)."""
    ctx = ctx or {}
    token_for = ctx.get("token_for")
    base_url = ctx.get("base_url")
    if not token_for or not base_url:
        return ""
    buttons = []
    styles = {
        "save": "background:#dcfce7;color:#166534;",
        "skip": "background:#fee2e2;color:#991b1b;",
    }
    labels = {"save": "👍 Save", "skip": "👎 Skip"}
    for action in ("save", "skip"):
        url = f"{base_url}/api/feedback/one-tap?token={token_for(paper.id, action, profile_id)}"
        buttons.append(
            f'<a href="{escape(url, quote=True)}" style="display:inline-block;{styles[action]}'
            "padding:8px 16px;border-radius:8px;font-size:13px;font-weight:600;"
            f'text-decoration:none;margin-right:8px;">{labels[action]}</a>'
        )
    return '<div style="margin-top:10px;">' + "".join(buttons) + "</div>"


def _figure_img(paper: Paper, ctx: dict | None) -> str:
    src = ((ctx or {}).get("figure_srcs") or {}).get(paper.id)
    if not src:
        return ""
    return (
        f'<img src="{escape(src, quote=True)}" alt="Figure from {escape(paper.title[:80], quote=True)}" '
        'width="560" style="display:block;width:100%;max-width:100%;height:auto;'
        'border:1px solid #e5e7eb;border-radius:8px;margin:0 0 10px;">'
    )


def _render_paper_html(
    paper: Paper, ctx: dict | None = None, *, hero: bool = False, profile_id: int | None = None
) -> str:
    """Render a single paper card as HTML with proper escaping.

    All styling is inline (email clients strip <style>); the layout is a single
    column so it reads well at phone width.
    """
    match_badges = "".join(
        f'<span style="display:inline-block;background:#e0e7ff;color:#3730a3;'
        f'padding:2px 8px;border-radius:12px;font-size:12px;margin-right:4px;">'
        f"{escape(t.strip())}</span>"
        for t in (paper.match_type or "").split("+")
        if t.strip()
    )

    topic_tags = ""
    if paper.topic_tags:
        tags = paper.topic_tags_list
        topic_tags = " ".join(
            f'<span style="display:inline-block;background:#f0fdf4;color:#166534;'
            f'padding:2px 6px;border-radius:8px;font-size:11px;margin-right:3px;">'
            f"{escape(t)}</span>"
            for t in tags[:6]
        )

    resource_links_html = ""
    for res in paper.resource_links_list:
        url = escape(res.get("url", ""), quote=True)
        label = escape(res.get("type", "link"))
        resource_links_html += f' <a href="{url}" style="color:#2563eb;font-size:12px;margin-right:6px;">[{label}]</a>'

    hero_banner = ""
    if hero:
        hero_banner = (
            '<div style="color:#7c3aed;font-size:11px;font-weight:700;letter-spacing:0.08em;'
            'text-transform:uppercase;margin-bottom:6px;">Top pick</div>'
        )
    title_size = "20px" if hero else "16px"
    border = "2px solid #c7d2fe" if hero else "1px solid #e5e7eb"

    return f"""
    <div style="border:{border};border-radius:12px;padding:16px;margin-bottom:14px;background:#ffffff;">
      {hero_banner}
      {_figure_img(paper, ctx)}
      <div style="margin-bottom:4px;">{match_badges}</div>
      <a href="{escape(paper.link, quote=True)}"
         style="color:#1d4ed8;font-size:{title_size};font-weight:600;text-decoration:none;">
        {escape(paper.title)}
      </a>
      <div style="color:#6b7280;font-size:13px;margin:4px 0;">
        {escape(paper.authors[:200])}
      </div>
      <div style="color:#374151;font-size:13px;line-height:1.5;margin:6px 0;">
        {escape(paper.summary_text or paper.abstract_text[:300])}
      </div>
      <div style="margin-top:8px;">
        {_score_chip(paper)}
        {topic_tags}
        {resource_links_html}
      </div>
      {_one_tap_buttons(paper, ctx, profile_id)}
    </div>
    """


def _render_alerts_html(alerts: list[dict]) -> str:
    """ "Saved search alerts" section: new matches per notify-flagged saved search."""
    if not alerts:
        return ""
    sections = []
    for entry in alerts:
        items = "".join(
            f'<li style="margin:4px 0;"><a href="{escape(p.link, quote=True)}" '
            f'style="color:#1d4ed8;font-size:13px;text-decoration:none;">{escape(p.title)}</a></li>'
            for p in entry["papers"]
        )
        sections.append(
            '<div style="margin-bottom:12px;">'
            f'<div style="font-size:13px;font-weight:600;color:#374151;">{escape(entry["name"])}</div>'
            f'<ul style="margin:4px 0 0;padding-left:18px;">{items}</ul></div>'
        )
    return (
        '<hr style="border:none;border-top:1px solid #e5e7eb;margin:20px 0 12px;">'
        '<h2 style="font-size:16px;color:#111827;margin:0 0 10px;">Saved search alerts</h2>' + "".join(sections)
    )


def _render_profile_section(section: dict, ctx: dict) -> str:
    """One digest section for a single interest profile (header + its cards)."""
    papers = section.get("papers") or []
    if not papers:
        return ""
    profile_id = section.get("profile_id")
    header = (
        '<h2 style="font-size:15px;color:#111827;margin:20px 0 10px;padding-top:6px;'
        f'border-top:1px solid #e5e7eb;">{escape(section.get("name") or "Papers")}</h2>'
    )
    cards = [_render_paper_html(papers[0], ctx, hero=True, profile_id=profile_id)]
    cards.extend(_render_paper_html(p, ctx, profile_id=profile_id) for p in papers[1:])
    return header + "\n".join(cards)


def _build_email_body(papers: list[Paper], today: date, ctx: dict | None = None) -> str:
    """Compose the full HTML email body (mobile-first, ≤600px single column).

    With 2+ digest-enabled interest profiles, ``ctx["sections"]`` drives one
    labeled section per profile (each ranked by that profile's model). The
    single-profile / default case keeps the original headerless hero layout.
    """
    ctx = ctx or {}
    sections = ctx.get("sections")
    if sections and len(sections) > 1:
        rendered = [html for html in (_render_profile_section(s, ctx) for s in sections) if html]
        paper_cards = "\n".join(rendered) or (
            '<p style="color:#6b7280;text-align:center;padding:40px 0;">No new matching papers found today.</p>'
        )
    elif not papers:
        paper_cards = (
            '<p style="color:#6b7280;text-align:center;padding:40px 0;">No new matching papers found today.</p>'
        )
    else:
        default_pid = ctx.get("default_profile_id")
        cards = [_render_paper_html(papers[0], ctx, hero=True, profile_id=default_pid)]
        cards.extend(_render_paper_html(p, ctx, profile_id=default_pid) for p in papers[1:])
        paper_cards = "\n".join(cards)

    catch_up_banner = ""
    if ctx.get("catch_up_label"):
        catch_up_banner = (
            '<div style="background:#fef3c7;color:#92400e;border-radius:8px;padding:10px 14px;'
            f'font-size:13px;font-weight:600;margin:0 0 16px;">{escape(ctx["catch_up_label"])}</div>'
        )

    synthesis_html = ""
    synthesis = ctx.get("synthesis")
    if synthesis and synthesis.get("topics"):
        if synthesis.get("narrative"):
            body = f'<p style="color:#374151;font-size:13px;margin:0;">{escape(synthesis["narrative"])}</p>'
        else:
            rows = "".join(
                f"<li>{escape(topic['label'])} &mdash; {int(topic['recent_count'])} new"
                f" (+{topic['delta_share']:.0%} share)</li>"
                for topic in synthesis["topics"]
            )
            body = f'<ul style="color:#374151;font-size:13px;margin:0;padding-left:18px;">{rows}</ul>'
        synthesis_html = (
            '<div style="background:#eef2ff;border-radius:10px;padding:12px 14px;margin:0 0 16px;">'
            '<div style="font-size:13px;font-weight:700;color:#3730a3;margin:0 0 6px;">'
            "📈 This week in your field</div>"
            f"{body}</div>"
        )

    exploration_html = ""
    exploration = ctx.get("exploration") or []
    if exploration:
        default_pid = ctx.get("default_profile_id")
        exploration_cards = "\n".join(_render_paper_html(p, ctx, profile_id=default_pid) for p in exploration)
        exploration_html = (
            '<h2 style="font-size:15px;color:#111827;margin:20px 0 2px;">🧭 Exploration</h2>'
            '<p style="color:#6b7280;font-size:12px;margin:0 0 10px;">'
            "Outside your usual lane — rating these teaches the ranker where your interests end.</p>"
            f"{exploration_cards}"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f9fafb;font-family:
  -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="max-width:600px;margin:0 auto;padding:12px;">
    <div style="background:white;border-radius:12px;padding:16px;
                box-shadow:0 1px 3px rgba(0,0,0,0.1);">
      <h1 style="font-size:22px;color:#111827;margin:0 0 4px;">
        ArXiv CV Digest
      </h1>
      <p style="color:#6b7280;font-size:14px;margin:0 0 16px;">
        {escape(today.strftime("%A, %B %d, %Y"))} &mdash;
        {len(papers)} paper{"s" if len(papers) != 1 else ""} matched
      </p>
      {catch_up_banner}
      {synthesis_html}
      {paper_cards}
      {exploration_html}
      {_render_alerts_html(ctx.get("alerts") or [])}
      <hr style="border:none;border-top:1px solid #e5e7eb;margin:20px 0 12px;">
      <p style="color:#9ca3af;font-size:11px;text-align:center;margin:0;">
        Sent by ArXiv CV Scraper &middot; 👍/👎 links train your ranking without opening the app
      </p>
    </div>
  </div>
</body>
</html>"""


def build_digest_mime(recipient: str, subject: str, html_body: str, attachments: list[tuple[str, bytes]]):
    """Assemble the outgoing message.

    With inline figures the structure is multipart/related (HTML first, then the
    CID-referenced PNG parts); without figures it stays the historical
    multipart/alternative with a single HTML part.
    """
    from email.mime.image import MIMEImage
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("related" if attachments else "alternative")
    msg["From"] = "me"
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))
    for cid, data in attachments:
        image = MIMEImage(data, _subtype="png")
        image.add_header("Content-ID", f"<{cid}>")
        image.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        msg.attach(image)
    return msg


def send_digest(app: Flask, *, dry_run: bool = False, force: bool = False) -> dict:
    """Query today's papers and send a digest via Gmail API.

    Returns a dict with keys: ``papers_count``, ``sent``, ``recipient`` (plus
    ``skipped_reason`` when the weekday schedule suppressed the send). Pass
    ``force=True`` (the manual "Send Test Digest" button) to ignore the schedule.
    """
    email_cfg = _get_email_config(app)
    digest_cfg = get_digest_config(app)

    recipient = email_cfg["recipient"]
    if not recipient:
        # Record the misconfiguration as an errored run before raising. A scheduled
        # digest trips this on every run, and without a DigestRun row the dashboard
        # and the Settings digest panel show nothing at all — the only trace ends up
        # in the server log, where nobody looks.
        message = "No recipient configured. Set 'email.recipient' in config.yaml."
        _finish_digest_run(
            app,
            _create_digest_run(app, recipient="", subject="", papers_count=0, preview_only=dry_run),
            status="error",
            error_message=message,
        )
        raise ValueError(message)

    if not force and not weekday_allowed(digest_cfg):
        LOGGER.info("Digest skipped: %s is not in the configured weekdays", utc_today().strftime("%A"))
        return {"papers_count": 0, "sent": False, "recipient": recipient, "skipped_reason": "weekday"}

    preview = build_digest_preview(app)
    subject = preview["subject"]
    papers = preview["papers"]
    digest_run_id = _create_digest_run(
        app,
        recipient=recipient,
        subject=subject,
        papers_count=len(papers),
        preview_only=dry_run,
    )

    # Re-render for email delivery: figures become CID attachments (mail clients
    # can't fetch localhost URLs) and one-tap tokens carry the digest run id.
    attachments, figure_srcs = _collect_figure_attachments(app, papers)
    ctx = {
        "base_url": digest_cfg["base_url"],
        "token_for": lambda paper_id, action, profile_id=None: make_one_tap_token(
            app, paper_id, action, digest_run_id, profile_id
        ),
        "figure_srcs": figure_srcs,
        "catch_up_label": preview["catch_up_label"],
        "alerts": preview["alerts"],
        "sections": preview.get("sections"),
        "default_profile_id": preview.get("default_profile_id"),
    }
    html_body = _build_email_body(papers, utc_today(), ctx)
    msg = build_digest_mime(recipient, subject, html_body, attachments)

    if dry_run:
        LOGGER.info("Dry run — email not sent (would send to %s)", recipient)
        _finish_digest_run(app, digest_run_id, status="preview")
        return {"papers_count": len(papers), "sent": False, "recipient": recipient}

    try:
        # Inside the try so a missing/expired token (the most common real-world
        # failure) is recorded as an errored DigestRun instead of leaving the row
        # stuck in "running" forever.
        creds = _load_gmail_credentials()
        service = _build_gmail_service(creds)
        raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
        service.users().messages().send(
            userId="me",
            body={"raw": raw_message},
        ).execute()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        # Credential/config failures already carry a contract the callers handle.
        _finish_digest_run(app, digest_run_id, status="error", error_message=str(exc))
        raise
    except Exception as exc:
        # Gmail API / auth send failures (HttpError, GoogleAuthError, network drops)
        # are plain Exception subclasses that callers don't catch. Translate them to
        # the RuntimeError contract both the settings route and the CLI already handle.
        _finish_digest_run(app, digest_run_id, status="error", error_message=str(exc))
        raise RuntimeError(f"Gmail API send failed: {exc}") from exc

    LOGGER.info("Digest sent to %s (%d papers)", recipient, len(papers))
    _finish_digest_run(app, digest_run_id, status="success")
    return {"papers_count": len(papers), "sent": True, "recipient": recipient}
