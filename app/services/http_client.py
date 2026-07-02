"""HTTP helpers with retry/backoff defaults."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from app.services.rate_limiter import get_shared_rate_limiter, resolve_rate_limit_settings

LOGGER = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "cv-arxiv-scraper/1.0"

# 4xx statuses that are still worth retrying: rate-limit (429), request timeout
# (408), too-early (425). Every other 4xx is a permanent client error — retrying
# a 404/403/400 just wastes a request and a backoff sleep, and (for arXiv PDFs of
# freshly announced papers) spams the log with pointless retries.
_RETRYABLE_4XX = frozenset({408, 425, 429})

# Upper bound on how long we'll honour a server-supplied ``Retry-After`` header.
# arXiv (and other origins) can send large or even hostile values; sleeping for
# minutes/hours would hang the worker, so clamp to something sane.
_MAX_RETRY_AFTER_SECONDS = 120.0

# Safety ceiling (bytes) on how much of a response body we'll buffer into the
# single worker's heap. Without a cap, a compromised/misbehaving origin (an arXiv
# mirror, or any host behind a user-configured feed URL) can trickle a
# multi-gigabyte or effectively unbounded body below the read timeout and OOM the
# process. PDFs are the largest legitimate payload, so the default is generous;
# callers can pass a tighter ``max_bytes`` (e.g. for feeds/XML).
_DEFAULT_MAX_BYTES = 200 * 1024 * 1024


class ResponseTooLargeError(Exception):
    """Raised when a response body exceeds the configured ``max_bytes`` ceiling."""


def _read_capped_body(response: requests.Response, max_bytes: int) -> None:
    """Buffer ``response`` body into memory, aborting once ``max_bytes`` is exceeded.

    Reads incrementally via :meth:`requests.Response.iter_content` and stashes the
    result into ``response._content`` so ``.content``/``.text``/``.json()`` keep
    working transparently for callers. Rejects early on a declared
    ``Content-Length`` over the cap to avoid downloading a body we'll discard.
    """
    headers = getattr(response, "headers", None)
    declared = headers.get("Content-Length") if headers else None
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                response.close()
                raise ResponseTooLargeError(f"Declared Content-Length {int(declared)} exceeds cap {max_bytes}")
        except (TypeError, ValueError):
            pass  # Malformed header — fall through to the streamed byte count guard.

    try:
        chunk_iter = iter(response.iter_content(chunk_size=64 * 1024))
    except TypeError:
        # The response object doesn't yield a real byte stream (e.g. a test
        # double). Nothing to cap — leave .content/.text untouched.
        return

    chunks: list[bytes] = []
    total = 0
    for chunk in chunk_iter:
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            response.close()
            raise ResponseTooLargeError(f"Response body exceeds cap {max_bytes}")
        chunks.append(chunk)
    # Populate the private cache requests uses so .content/.text/.json() are served
    # from our capped buffer instead of re-reading the (already consumed) stream.
    response._content = b"".join(chunks)
    response._content_consumed = True


def _parse_retry_after(exc: Exception) -> float | None:
    """Extract a ``Retry-After`` delay (seconds) from an HTTP error response.

    Honours both supported forms — an integer number of seconds and an
    HTTP-date — and clamps the result to ``[0, _MAX_RETRY_AFTER_SECONDS]``.
    Returns ``None`` when the response has no usable header so the caller can
    fall back to its computed backoff.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None

    seconds: float | None = None
    try:
        seconds = float(int(raw))
    except (TypeError, ValueError):
        try:
            parsed = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            seconds = (parsed - datetime.now(timezone.utc)).total_seconds()

    if seconds is None:
        return None
    return max(0.0, min(seconds, _MAX_RETRY_AFTER_SECONDS))


def _is_retryable(exc: Exception) -> bool:
    """Whether ``exc`` from an HTTP attempt is worth retrying.

    Network-level failures (timeouts, connection resets) carry no response and
    are transient, so retry them. HTTP errors are retried only for 5xx and the
    select 4xx in :data:`_RETRYABLE_4XX`; all other 4xx fail fast.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        return True
    return status >= 500 or status in _RETRYABLE_4XX


_SESSION_LIMITER_ATTR = "_cv_arxiv_rate_limiter"
_SESSION_RATE_LIMIT_ATTR = "_cv_arxiv_rate_limit_settings"
_SESSION_USER_AGENT_ATTR = "_cv_arxiv_user_agent"


def resolve_user_agent(scraper_config: Mapping[str, Any] | None = None, *, fallback: str = DEFAULT_USER_AGENT) -> str:
    """Resolve the outgoing HTTP User-Agent."""
    ingest = scraper_config.get("ingest", {}) if isinstance(scraper_config, Mapping) else {}
    user_agent = ingest.get("user_agent") if isinstance(ingest, Mapping) else None
    if isinstance(user_agent, str) and user_agent.strip():
        return user_agent.strip()
    return fallback


def _headers_with_user_agent(headers: Mapping[str, str] | None, *, user_agent: str) -> dict[str, str]:
    merged = dict(headers or {})
    if not any(key.lower() == "user-agent" for key in merged):
        merged["User-Agent"] = user_agent
    return merged


def _apply_session_config(session: requests.Session, *, settings, user_agent: str):
    """Stamp resolved rate-limit settings + User-Agent onto a session."""
    limiter = get_shared_rate_limiter(settings)
    session.headers["User-Agent"] = user_agent
    setattr(session, _SESSION_LIMITER_ATTR, limiter)
    setattr(session, _SESSION_RATE_LIMIT_ATTR, settings)
    setattr(session, _SESSION_USER_AGENT_ATTR, user_agent)
    return limiter


def _configure_session(
    session: requests.Session,
    *,
    scraper_config: Mapping[str, Any] | None = None,
    rate_limit_profile: str = "interactive",
    user_agent: str | None = None,
) -> None:
    effective_user_agent = user_agent or resolve_user_agent(scraper_config)
    effective_settings = resolve_rate_limit_settings(scraper_config, profile=rate_limit_profile)
    _apply_session_config(session, settings=effective_settings, user_agent=effective_user_agent)


def request_with_backoff(
    method: str,
    url: str,
    *,
    attempts: int = 3,
    base_delay: float = 1.25,
    timeout: int = 30,
    session: requests.Session | None = None,
    scraper_config: Mapping[str, Any] | None = None,
    rate_limit_profile: str | None = None,
    user_agent: str | None = None,
    max_bytes: int | None = _DEFAULT_MAX_BYTES,
    **kwargs: Any,
) -> requests.Response:
    """Run an HTTP request with bounded retries and exponential backoff.

    Response bodies are streamed and buffered up to ``max_bytes`` (default
    :data:`_DEFAULT_MAX_BYTES`); a body exceeding the cap raises
    :class:`ResponseTooLargeError`. Pass ``max_bytes=None`` to disable the cap.
    The buffered body is cached on the response, so callers keep using
    ``.content``/``.text``/``.json()`` unchanged.
    """
    # Always make at least one attempt. A misconfigured ``attempts <= 0`` (e.g.
    # ``pdf_attempts: 0``) would otherwise skip the loop entirely and ``raise
    # last_exc`` with last_exc still None → confusing ``TypeError``, no request made.
    attempts = max(1, attempts)
    last_exc: Exception | None = None
    requested_settings = resolve_rate_limit_settings(scraper_config, profile=rate_limit_profile or "interactive")
    requested_user_agent = user_agent or resolve_user_agent(scraper_config)
    if session is not None:
        limiter = getattr(session, _SESSION_LIMITER_ATTR, None)
        session_settings = getattr(session, _SESSION_RATE_LIMIT_ATTR, None)
        effective_user_agent = getattr(session, _SESSION_USER_AGENT_ATTR, None)
        # Reconfigure each dimension only when the caller actually specified it; for
        # the rest, inherit the session's existing configuration rather than reset it
        # to library defaults. A configured scrape session is reused across calls that
        # specify different subsets: the RSS/PDF backends pass only ``session=`` (must
        # keep both the configured User-Agent and rate limit), while the enrichment /
        # arxiv_api backends pass an explicit ``rate_limit_profile="bulk"`` (must
        # re-tune the throttle) — and sometimes ``user_agent`` — but no scraper_config.
        # Without per-dimension intent, the first bare call clobbered a configured
        # ``ingest.user_agent`` (and a non-default profile call wiped the rate limit).
        unconfigured = limiter is None or session_settings is None or effective_user_agent is None
        wants_settings = unconfigured or scraper_config is not None or rate_limit_profile is not None
        wants_user_agent = unconfigured or user_agent is not None or scraper_config is not None
        target_settings = requested_settings if wants_settings else session_settings
        target_user_agent = requested_user_agent if wants_user_agent else effective_user_agent
        if unconfigured or session_settings != target_settings or effective_user_agent != target_user_agent:
            limiter = _apply_session_config(session, settings=target_settings, user_agent=target_user_agent)
            effective_user_agent = target_user_agent
        do_request = session.request
        request_headers = kwargs.get("headers")
        if request_headers is not None:
            kwargs["headers"] = _headers_with_user_agent(request_headers, user_agent=effective_user_agent)
    else:
        effective_user_agent = requested_user_agent
        limiter = get_shared_rate_limiter(requested_settings)
        kwargs["headers"] = _headers_with_user_agent(kwargs.get("headers"), user_agent=effective_user_agent)
        do_request = requests.request

    # Stream so we can enforce a total-bytes ceiling as the body arrives, rather
    # than letting requests buffer an unbounded body into the worker's heap. When
    # the caller opts out (max_bytes=None) or already requested streaming, respect
    # their choice and skip the capped read.
    stream_for_cap = max_bytes is not None and "stream" not in kwargs
    if stream_for_cap:
        kwargs["stream"] = True

    for attempt in range(1, attempts + 1):
        try:
            limiter.acquire()
            response = do_request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            if stream_for_cap:
                _read_capped_body(response, max_bytes)
            return response
        except ResponseTooLargeError:
            # Retrying just re-downloads the same oversized body — fail fast.
            raise
        except Exception as exc:  # pragma: no cover - exercised by integration paths
            last_exc = exc
            if attempt == attempts or not _is_retryable(exc):
                break

            delay = base_delay * (2 ** (attempt - 1))
            retry_after = _parse_retry_after(exc)
            if retry_after is not None:
                delay = max(delay, retry_after)
            LOGGER.warning(
                "HTTP retry %s/%s for %s %s after error: %s",
                attempt,
                attempts,
                method,
                url,
                exc,
            )
            time.sleep(delay)

    raise last_exc  # guaranteed non-None by loop logic


def create_session(
    pool_size: int = 10,
    *,
    scraper_config: Mapping[str, Any] | None = None,
    rate_limit_profile: str = "interactive",
    user_agent: str | None = None,
) -> requests.Session:
    """Create a session with connection pooling for concurrent downloads."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _configure_session(
        session,
        scraper_config=scraper_config,
        rate_limit_profile=rate_limit_profile,
        user_agent=user_agent,
    )
    return session
