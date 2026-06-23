"""Zotero API client for syncing papers.

Uses API key authentication (simpler than OAuth2).
Credentials are stored in ``.zotero_credentials`` (JSON with ``api_key``
and ``user_id``).
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

LOGGER = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CREDENTIALS_PATH = _PROJECT_ROOT / ".zotero_credentials"

ZOTERO_API_BASE = "https://api.zotero.org"
ZOTERO_BATCH_LIMIT = 50


class ZoteroClient:
    """Client for interacting with the Zotero API."""

    def __init__(self, credentials_path: Path | None = None):
        self.credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH
        self._creds: dict | None = None

    def _load_credentials(self) -> dict:
        """Load api_key and user_id from credentials file."""
        if self._creds:
            return self._creds
        data = json.loads(self.credentials_path.read_text(encoding="utf-8"))
        self._creds = {
            "api_key": data["api_key"],
            "user_id": data["user_id"],
        }
        return self._creds

    def _save_credentials(self, api_key: str, user_id: str) -> None:
        """Save credentials to disk with restricted permissions."""
        data = {"api_key": api_key, "user_id": user_id}
        self.credentials_path.write_text(json.dumps(data), encoding="utf-8")
        os.chmod(self.credentials_path, 0o600)
        self._creds = data

    def _get_headers(self) -> dict:
        """Get authorization headers."""
        creds = self._load_credentials()
        return {
            "Zotero-API-Key": creds["api_key"],
            "Content-Type": "application/json",
        }

    def _user_url(self) -> str:
        """Get the base URL for the user's library."""
        creds = self._load_credentials()
        return f"{ZOTERO_API_BASE}/users/{creds['user_id']}"

    def check_connection(self) -> dict:
        """Verify API key validity and return status dict.

        Returns same shape as ``check_gmail_auth_status()``.
        """
        if not self.credentials_path.exists():
            return {
                "status": "no_credentials",
                "message": ("Zotero not configured. Enter your API key and user ID from zotero.org/settings/keys."),
            }

        try:
            self._load_credentials()
        except Exception:
            return {
                "status": "invalid",
                "message": "Zotero credentials file is corrupted.",
            }

        try:
            resp = requests.get(
                f"{self._user_url()}/items/top",
                headers=self._get_headers(),
                params={"limit": 1},
                timeout=10,
            )
            if resp.status_code == 200:
                return {
                    "status": "connected",
                    "message": "Zotero is connected.",
                }
            elif resp.status_code in (401, 403):
                return {
                    "status": "invalid",
                    "message": "Zotero API key is invalid or expired. Check your key.",
                }
            else:
                return {
                    "status": "error",
                    "message": f"Zotero API returned status {resp.status_code}.",
                }
        except requests.RequestException as exc:
            return {
                "status": "error",
                "message": f"Could not reach Zotero API: {exc}",
            }

    def list_collections(self) -> list[dict]:
        """List the user's Zotero collections.

        Returns list of dicts with ``key`` and ``name``.
        """
        try:
            resp = requests.get(
                f"{self._user_url()}/collections",
                headers=self._get_headers(),
                timeout=10,
            )
            resp.raise_for_status()
            return [{"key": c["key"], "name": c["data"]["name"]} for c in resp.json()]
        except requests.RequestException:
            return []

    def _paper_to_zotero_item(self, paper: Paper, collection_key: str | None = None) -> dict:
        """Map Paper fields to Zotero journalArticle item type."""
        creators = []
        for name in paper.authors.split(","):
            name = name.strip()
            if not name:
                continue
            parts = name.rsplit(None, 1)
            if len(parts) == 2:
                creators.append(
                    {
                        "creatorType": "author",
                        "firstName": parts[0],
                        "lastName": parts[1],
                    }
                )
            else:
                creators.append(
                    {
                        "creatorType": "author",
                        "name": name,
                    }
                )

        item: dict = {
            "itemType": "journalArticle",
            "title": paper.title,
            "creators": creators,
            "abstractNote": paper.abstract_text or "",
            "url": paper.link,
        }

        if paper.publication_dt:
            item["date"] = paper.publication_dt.isoformat()

        if paper.arxiv_id:
            item["extra"] = f"arXiv:{paper.arxiv_id}"

        if collection_key:
            item["collections"] = [collection_key]

        return item

    @staticmethod
    def _failed_items(resp: requests.Response) -> dict:
        """Extract Zotero's per-item ``failed`` map from a 200 response.

        The Zotero Web API returns HTTP 200 even when individual items are
        rejected (validation, quota, …), reporting the outcome in the body as
        ``{"successful": …, "unchanged": …, "failed": {...}}``. Treating any 200
        as full success silently loses those items, so callers must inspect this.
        Defensive against non-JSON / non-dict bodies (older/mocked responses).
        """
        try:
            body = resp.json()
        except (ValueError, TypeError):
            return {}
        if isinstance(body, dict) and isinstance(body.get("failed"), dict):
            return body["failed"]
        return {}

    def add_item(self, paper: Paper, collection_key: str | None = None) -> dict:
        """Add a paper to the user's Zotero library.

        Returns dict with ``success``, ``message``.
        """
        item = self._paper_to_zotero_item(paper, collection_key)

        try:
            resp = requests.post(
                f"{self._user_url()}/items",
                headers=self._get_headers(),
                json=[item],
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            return {"success": False, "message": f"Failed to add item: {exc}"}

        failed = self._failed_items(resp)
        if failed:
            reason: object = next(iter(failed.values()), {})
            message = reason.get("message") if isinstance(reason, dict) else None
            return {"success": False, "message": f"Zotero rejected the item: {message or 'unknown error'}"}
        return {"success": True, "message": "Item added to Zotero."}

    def sync_saved_papers(
        self,
        papers: list[Paper],
        collection_key: str | None = None,
    ) -> dict:
        """Batch sync papers to Zotero (max 50 per request).

        Returns dict with ``success``, ``message``, ``synced_count``.
        """
        items = [self._paper_to_zotero_item(p, collection_key) for p in papers]
        synced = 0
        rejected = 0

        for i in range(0, len(items), ZOTERO_BATCH_LIMIT):
            batch = items[i : i + ZOTERO_BATCH_LIMIT]
            try:
                resp = requests.post(
                    f"{self._user_url()}/items",
                    headers=self._get_headers(),
                    json=batch,
                    timeout=60,
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                return {
                    "success": False,
                    "message": f"Sync failed after {synced} items: {exc}",
                    "synced_count": synced,
                }
            # A 200 can still reject individual items; only count the accepted ones.
            failed = len(self._failed_items(resp))
            synced += len(batch) - failed
            rejected += failed

        if rejected:
            return {
                "success": False,
                "message": f"Synced {synced} of {len(items)} papers; Zotero rejected {rejected}.",
                "synced_count": synced,
            }
        return {
            "success": True,
            "message": f"Synced {synced} papers to Zotero.",
            "synced_count": synced,
        }
