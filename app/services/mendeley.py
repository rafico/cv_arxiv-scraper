"""Mendeley API client for syncing papers.

Uses OAuth2 for authentication, mirroring the Gmail OAuth pattern.
Credentials are stored in ``mendeley_credentials.json`` and tokens in
``.mendeley_token``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from app.models import Paper

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CREDENTIALS_PATH = _PROJECT_ROOT / "mendeley_credentials.json"
DEFAULT_TOKEN_PATH = _PROJECT_ROOT / ".mendeley_token"

MENDELEY_AUTH_URL = "https://api.mendeley.com/oauth/authorize"
MENDELEY_TOKEN_URL = "https://api.mendeley.com/oauth/token"
MENDELEY_API_BASE = "https://api.mendeley.com"


class MendeleyClient:
    """Client for interacting with the Mendeley API."""

    def __init__(
        self,
        credentials_path: Path | None = None,
        token_path: Path | None = None,
    ):
        self.credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH
        self.token_path = token_path or DEFAULT_TOKEN_PATH
        self._token_data: dict | None = None

    def _load_credentials(self) -> dict:
        """Load client_id and client_secret from credentials file."""
        data = json.loads(self.credentials_path.read_text(encoding="utf-8"))
        return {
            "client_id": data["client_id"],
            "client_secret": data["client_secret"],
        }

    def _load_token(self) -> dict:
        """Load the stored OAuth token."""
        if self._token_data:
            return self._token_data
        data = json.loads(self.token_path.read_text(encoding="utf-8"))
        self._token_data = data
        return data

    def _save_token(self, token_data: dict) -> None:
        """Save OAuth token to disk with restricted permissions."""
        self.token_path.write_text(json.dumps(token_data), encoding="utf-8")
        os.chmod(self.token_path, 0o600)
        self._token_data = token_data

    def _get_headers(self) -> dict:
        """Get authorization headers using the stored token."""
        token = self._load_token()
        return {
            "Authorization": f"Bearer {token['access_token']}",
            "Content-Type": "application/vnd.mendeley-document.1+json",
        }

    def check_connection(self) -> dict:
        """Verify token validity and return status dict.

        Returns same shape as ``check_gmail_auth_status()``.
        """
        if not self.credentials_path.exists():
            return {
                "status": "no_credentials",
                "message": "Mendeley credentials not found. Upload mendeley_credentials.json.",
            }

        if not self.token_path.exists():
            return {
                "status": "no_token",
                "message": "Mendeley not authorized. Click 'Authorize Mendeley' to connect.",
            }

        try:
            token = self._load_token()
        except Exception:
            return {
                "status": "invalid",
                "message": "Mendeley token is corrupted. Re-authorize.",
            }

        try:
            resp = requests.get(
                f"{MENDELEY_API_BASE}/profiles/me",
                headers={"Authorization": f"Bearer {token['access_token']}"},
                timeout=10,
            )
            if resp.status_code == 200:
                return {
                    "status": "connected",
                    "message": "Mendeley is connected.",
                }
            elif resp.status_code == 401:
                return {
                    "status": "expired",
                    "message": "Mendeley token expired. Re-authorize to reconnect.",
                }
            else:
                return {
                    "status": "invalid",
                    "message": f"Mendeley API returned status {resp.status_code}.",
                }
        except requests.RequestException as exc:
            return {
                "status": "error",
                "message": f"Could not reach Mendeley API: {exc}",
            }

    def start_oauth_flow(self, redirect_uri: str) -> dict:
        """Build Mendeley OAuth2 authorization URL.

        Returns dict with ``success``, ``auth_url``, ``state``, ``message``.
        """
        if not self.credentials_path.exists():
            return {
                "success": False,
                "auth_url": None,
                "state": None,
                "message": "mendeley_credentials.json not found.",
            }

        try:
            creds = self._load_credentials()
        except Exception as exc:
            return {
                "success": False,
                "auth_url": None,
                "state": None,
                "message": f"Invalid mendeley_credentials.json: {exc}",
            }

        from secrets import token_urlsafe
        state = token_urlsafe(32)

        params = {
            "client_id": creds["client_id"],
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "all",
            "state": state,
        }
        auth_url = f"{MENDELEY_AUTH_URL}?" + "&".join(
            f"{k}={requests.utils.quote(str(v))}" for k, v in params.items()
        )

        return {
            "success": True,
            "auth_url": auth_url,
            "state": state,
            "message": "Redirecting to Mendeley for authorization.",
        }

    def finish_oauth_flow(self, authorization_response_url: str, redirect_uri: str) -> dict:
        """Exchange the authorization code for an access token.

        Returns dict with ``success`` and ``message``.
        """
        from urllib.parse import parse_qs, urlparse

        try:
            creds = self._load_credentials()
        except Exception as exc:
            return {"success": False, "message": f"Invalid credentials: {exc}"}

        parsed = urlparse(authorization_response_url)
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        if not code:
            return {"success": False, "message": "No authorization code in callback URL."}

        try:
            resp = requests.post(
                MENDELEY_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": creds["client_id"],
                    "client_secret": creds["client_secret"],
                },
                timeout=30,
            )
            resp.raise_for_status()
            token_data = resp.json()
            self._save_token(token_data)
            return {"success": True, "message": "Mendeley authorized successfully."}
        except Exception as exc:
            return {"success": False, "message": f"Mendeley OAuth failed: {exc}"}

    def add_document(self, paper: Paper) -> dict:
        """Add a paper to the user's Mendeley library.

        Returns dict with ``success``, ``message``, and optionally ``document_id``.
        """
        doc = {
            "type": "journal",
            "title": paper.title,
            "authors": [
                {"first_name": parts[0], "last_name": parts[-1]}
                for name in paper.authors.split(",")
                if (parts := name.strip().rsplit(None, 1)) and len(parts) >= 1
            ],
            "identifiers": {},
            "websites": [paper.link],
        }

        if paper.arxiv_id:
            doc["identifiers"]["arxiv"] = paper.arxiv_id

        if paper.abstract_text:
            doc["abstract"] = paper.abstract_text

        if paper.publication_dt:
            doc["year"] = paper.publication_dt.year
            doc["month"] = paper.publication_dt.month

        try:
            resp = requests.post(
                f"{MENDELEY_API_BASE}/documents",
                headers=self._get_headers(),
                json=doc,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            return {
                "success": True,
                "message": "Document added to Mendeley.",
                "document_id": result.get("id"),
            }
        except requests.RequestException as exc:
            return {"success": False, "message": f"Failed to add document: {exc}"}
