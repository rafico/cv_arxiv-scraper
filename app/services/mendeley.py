"""Mendeley API client for syncing papers.

Uses OAuth2 for authentication, mirroring the Gmail OAuth pattern.
Credentials are stored in ``mendeley_credentials.json`` and tokens in
``.mendeley_token``.
"""

from __future__ import annotations

import json
import logging
import textwrap
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from app.services.secret_files import write_secret_file

if TYPE_CHECKING:
    from app.models import Collection, Paper

LOGGER = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CREDENTIALS_PATH = _PROJECT_ROOT / "mendeley_credentials.json"
DEFAULT_TOKEN_PATH = _PROJECT_ROOT / ".mendeley_token"

MENDELEY_AUTH_URL = "https://api.mendeley.com/oauth/authorize"
MENDELEY_TOKEN_URL = "https://api.mendeley.com/oauth/token"  # noqa: S105 — OAuth endpoint URL, not a secret
MENDELEY_API_BASE = "https://api.mendeley.com"

# Papers per export_collection call (up to two API calls each): small enough for the
# collection page to show progress between calls.
EXPORT_BATCH_SIZE = 10

# Held by every route that creates Mendeley documents (the per-paper send, the saved-papers
# sync, a collection's export batch) from its read of papers.mendeley_doc_id to its commit,
# so that none of them creates a document another one is just creating. Process-local,
# which is enough for this single-worker app.
WRITE_LOCK = threading.Lock()


def _explain(exc: Exception) -> str:
    """``exc`` and, for a refused request, Mendeley's answer: the reason is in the body, which requests leaves out."""
    response = getattr(exc, "response", None)
    body = response.text.strip()[:200] if response is not None else ""
    return f"{exc} ({body})" if body else str(exc)


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

    def _save_credentials(self, client_id: str, client_secret: str) -> None:
        """Save client_id and client_secret with restricted permissions."""
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
        }
        write_secret_file(self.credentials_path, json.dumps(data))

    def _load_token(self) -> dict:
        """Load the stored OAuth token."""
        if self._token_data:
            return self._token_data
        data = json.loads(self.token_path.read_text(encoding="utf-8"))
        if not data.get("access_token"):
            # A valid-JSON token file lacking access_token is unusable. Raise here
            # so check_connection()'s broad except converts it to a graceful
            # "invalid" status instead of a later KeyError → HTTP 500.
            raise ValueError("Stored Mendeley token has no access_token.")
        self._token_data = data
        return data

    def _save_token(self, token_data: dict) -> None:
        """Save OAuth token to disk with restricted permissions."""
        write_secret_file(self.token_path, json.dumps(token_data))
        self._token_data = token_data

    def _request(self, method: str, url: str, media: str = "document", **kwargs) -> requests.Response:
        """Authorized API call that raises on an error status.

        A 401 refreshes the token and retries once. ``media`` names the versioned
        Mendeley media type, sent as Accept on a GET and as Content-Type on a write.
        """
        # ponytail: one connection per call; share a requests.Session (and re-point the
        # tests' requests.get/post patches) if large first exports feel slow.
        send = getattr(requests, method)
        media_header = "Accept" if method == "get" else "Content-Type"

        def attempt() -> requests.Response:
            headers = {
                "Authorization": f"Bearer {self._load_token()['access_token']}",
                media_header: f"application/vnd.mendeley-{media}.1+json",
            }
            return send(url, headers=headers, timeout=30, **kwargs)

        resp = attempt()
        if resp.status_code == 401:
            try:
                self._refresh_access_token()
            except requests.HTTPError as exc:
                # The token endpoint's 4xx is an auth failure, not a verdict on this request.
                raise RuntimeError(f"Mendeley token could not be refreshed: {exc}") from exc
            resp = attempt()
        resp.raise_for_status()
        return resp

    def _paged(self, path: str, media: str = "document", **params: str) -> Iterator[dict]:
        """Every item of a paginated collection, 500 (the API's maximum) per request."""
        url: str | None = f"{MENDELEY_API_BASE}{path}"
        query: dict | None = {"limit": 500, **params}
        while url:
            resp = self._request("get", url, media, params=query)
            yield from resp.json()
            # The API tells clients to follow its Link URLs, not to build the next page's.
            url, query = resp.links.get("next", {}).get("url"), None
            if url and not url.startswith(f"{MENDELEY_API_BASE}/"):
                # The bearer token goes wherever this URL points.
                raise ValueError(f"Mendeley pagination link leaves the API: {url}")

    def _refresh_access_token(self) -> dict:
        """Refresh the stored OAuth token using the refresh token."""
        token = self._load_token()
        refresh_token = token.get("refresh_token")
        if not refresh_token:
            raise RuntimeError("Mendeley token expired and cannot be refreshed automatically.")

        creds = self._load_credentials()
        resp = requests.post(
            MENDELEY_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": creds["client_id"],
                "client_secret": creds["client_secret"],
            },
            timeout=5,
        )
        resp.raise_for_status()
        refreshed = resp.json()
        if not refreshed.get("access_token"):
            # A 2xx body without access_token is unusable; reject it so callers
            # (check_connection, add_document) surface a graceful status instead
            # of a KeyError when dereferencing refreshed['access_token'].
            raise ValueError("Mendeley token refresh response has no access_token.")
        if not refreshed.get("refresh_token"):
            refreshed["refresh_token"] = refresh_token
        self._save_token(refreshed)
        return refreshed

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
                timeout=3,
            )
            if resp.status_code == 401:
                try:
                    refreshed = self._refresh_access_token()
                except Exception:
                    return {
                        "status": "expired",
                        "message": "Mendeley token expired. Re-authorize to reconnect.",
                    }
                resp = requests.get(
                    f"{MENDELEY_API_BASE}/profiles/me",
                    headers={"Authorization": f"Bearer {refreshed['access_token']}"},
                    timeout=3,
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
        auth_url = f"{MENDELEY_AUTH_URL}?" + "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())

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
            if not token_data.get("access_token"):
                raise ValueError("Mendeley token response has no access_token.")
            self._save_token(token_data)
            return {"success": True, "message": "Mendeley authorized successfully."}
        except Exception as exc:
            return {"success": False, "message": f"Mendeley OAuth failed: {exc}"}

    def _create_document(self, paper: Paper) -> dict:
        """Create a library document for ``paper`` and return it as the API does; raises on failure."""
        doc = {
            "type": "journal",
            "title": paper.title,
            "authors": [
                # Single-token names (mononyms, collaborations like "ATLAS") have no
                # first name; emit an empty one rather than duplicating the token into
                # both fields.
                {"first_name": parts[0] if len(parts) > 1 else "", "last_name": parts[-1]}
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

        return self._request("post", f"{MENDELEY_API_BASE}/documents", json=doc).json()

    def add_document(self, paper: Paper) -> dict:
        """Add a paper to the user's Mendeley library.

        Returns dict with ``success``, ``message``, and optionally ``document_id``.
        """
        try:
            return {
                "success": True,
                "message": "Document added to Mendeley.",
                "document_id": self._create_document(paper).get("id"),
            }
        except (requests.RequestException, RuntimeError, ValueError, KeyError, OSError) as exc:
            # A mid-batch 401 triggers _refresh_access_token, which raises RuntimeError
            # when the stored token has no refresh_token (and KeyError/ValueError/OSError
            # on malformed creds). Those aren't RequestExceptions, so without this the
            # error escaped → caller 500s and an in-progress sync aborted before commit,
            # losing the mendeley_doc_id of already-synced papers (duplicate re-uploads).
            return {"success": False, "message": f"Failed to add document: {exc}"}

    def export_collection(self, collection: Collection, after: int = 0, limit: int = EXPORT_BATCH_SIZE) -> dict:
        """Send one batch of a collection's papers to its Mendeley folder; the caller holds ``WRITE_LOCK``.

        The folder is the one remembered in ``mendeley_folder_id`` while Mendeley still has
        it, else a library folder named like the collection that no other collection uses,
        else a new one. Safe to repeat and to resume: papers already in the folder are
        skipped, and a paper the library already holds (same arXiv id) is linked instead of
        duplicated. Nothing is ever removed from Mendeley.

        ``after`` is the last paper id the previous call tried, so that one pass over the
        collection ends even when some papers keep failing. Returns ``total``, ``sent``,
        ``failures`` (one message per paper Mendeley rejected), ``remaining`` and the next
        ``after`` (``None`` once the pass is over). When Mendeley as a whole is failing
        (network, auth, rate limit, server errors) it returns ``error`` with this call's
        ``sent``; what was sent stays sent.
        """
        from app.models import Collection, Paper, PaperCollection, db, in_review_clause
        from app.services.onboarding import normalize_arxiv_id

        sent = 0
        try:
            # Checked against the listing on every call rather than trusted: the API does not
            # say what it answers for the id of a folder that was deleted in Mendeley.
            folders = {folder["id"]: folder.get("name") or "" for folder in self._paged("/folders", "folder")}
            folder_id = collection.mendeley_folder_id
            if folder_id not in folders:  # first export, or the folder is gone
                # A folder of this name is taken over, unless it is another collection's: that
                # one may have been renamed since, and its old name given to this one.
                taken = {
                    fid
                    for (fid,) in db.session.query(Collection.mendeley_folder_id).filter(Collection.id != collection.id)
                }
                name = collection.name.strip().casefold()
                folder_id = next(
                    (fid for fid, theirs in folders.items() if fid not in taken and theirs.strip().casefold() == name),
                    None,
                )
                if folder_id is None:
                    created = self._request(
                        "post", f"{MENDELEY_API_BASE}/folders", "folder", json={"name": collection.name}
                    )
                    folder_id = created.json()["id"]
                collection.mendeley_folder_id = folder_id
                db.session.commit()
            # Trashed documents still belong to their folder. Counting them in leaves a paper
            # that was thrown away in Mendeley alone instead of sending it again.
            in_folder = {doc["id"] for doc in self._paged(f"/folders/{folder_id}/documents", include_trashed="true")}

            # The papers "Export .bib" exports: the review's visible ones.
            papers = (
                Paper.query.join(PaperCollection, PaperCollection.paper_id == Paper.id)
                .filter(PaperCollection.collection_id == collection.id, Paper.is_hidden.is_(False), in_review_clause())
                .order_by(Paper.id)
                .all()
            )
            if not after and any(not paper.mendeley_doc_id for paper in papers):
                # Start of a pass: link what the library already holds instead of duplicating it.
                library: dict[str, str] = {}
                for doc in self._paged("/documents"):
                    arxiv_id = normalize_arxiv_id((doc.get("identifiers") or {}).get("arxiv"))
                    if arxiv_id:
                        library.setdefault(arxiv_id, doc["id"])
                for paper in papers:
                    if not paper.mendeley_doc_id:
                        paper.mendeley_doc_id = library.get(paper.arxiv_id)

            pending = [paper for paper in papers if paper.id > after and paper.mendeley_doc_id not in in_folder]
            batch, rest = pending[:limit], pending[limit:]
            failures: list[str] = []
            shaky = False  # the paper before this one drew a 403 or a 5xx
            for paper in batch:
                adding = False
                try:
                    if not paper.mendeley_doc_id:
                        # ponytail: two calls per new paper. The API declares folder_uuids on
                        # POST /documents, which would file a new document in one call: use it
                        # once that is checked against the live API.
                        paper.mendeley_doc_id = self._create_document(paper)["id"]
                        # At once: a document whose id is lost here is created again next time.
                        db.session.commit()
                    adding = True
                    self._request(
                        "post", f"{MENDELEY_API_BASE}/folders/{folder_id}/documents", json={"id": paper.mendeley_doc_id}
                    )
                    sent += 1
                    shaky = False
                except requests.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else 0
                    if adding and status == 409:  # "document already exists in folder"
                        continue
                    message = f"{textwrap.shorten(paper.title, 80, placeholder='…')}: {_explain(exc)}"
                    unclear = status == 403 or not 400 <= status < 500
                    # 401 and 429 are never one paper's fault. A 403 or a 5xx may be (the API does
                    # not say), so the first is put down to its paper and a second in a row to
                    # Mendeley: stop then, rather than fail every paper after it.
                    if status in (401, 429) or (unclear and shaky):
                        raise RuntimeError(message) from exc
                    shaky = unclear
                    if adding and status == 404:
                        # "Folder or document not found", and the folder was listed a moment ago:
                        # the linked document is gone from this library (deleted for good, or it
                        # belonged to another account). Forget it, so that the next pass links or
                        # creates the paper afresh.
                        paper.mendeley_doc_id = None
                        message += " (its Mendeley document is gone: the next run sends the paper again)"
                    failures.append(message)
            db.session.commit()  # links to documents the library already held, and links dropped above
            return {
                "total": len(papers),
                "sent": sent,
                "failures": failures,
                "remaining": len(rest),
                "after": batch[-1].id if rest else None,
            }
        except (requests.RequestException, RuntimeError, ValueError, KeyError, OSError) as exc:
            db.session.rollback()
            return {"error": f"Mendeley export stopped: {_explain(exc)}", "sent": sent}
