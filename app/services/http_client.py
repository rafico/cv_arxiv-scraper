"""HTTP helpers with retry/backoff defaults."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import requests

from app.services.rate_limiter import get_shared_rate_limiter, resolve_rate_limit_settings

LOGGER = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "cv-arxiv-scraper/1.0"
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


def _configure_session(
    session: requests.Session,
    *,
    scraper_config: Mapping[str, Any] | None = None,
    rate_limit_profile: str = "interactive",
    user_agent: str | None = None,
):
    effective_user_agent = user_agent or resolve_user_agent(scraper_config)
    effective_settings = resolve_rate_limit_settings(scraper_config, profile=rate_limit_profile)
    limiter = get_shared_rate_limiter(effective_settings)

    session.headers["User-Agent"] = effective_user_agent
    setattr(session, _SESSION_LIMITER_ATTR, limiter)
    setattr(session, _SESSION_RATE_LIMIT_ATTR, effective_settings)
    setattr(session, _SESSION_USER_AGENT_ATTR, effective_user_agent)
    return limiter, effective_user_agent


def request_with_backoff(
    method: str,
    url: str,
    *,
    attempts: int = 3,
    base_delay: float = 1.25,
    timeout: int = 30,
    session: requests.Session | None = None,
    scraper_config: Mapping[str, Any] | None = None,
    rate_limit_profile: str = "interactive",
    user_agent: str | None = None,
    **kwargs: Any,
) -> requests.Response:
    """Run an HTTP request with bounded retries and exponential backoff."""
    last_exc: Exception | None = None
    requested_settings = resolve_rate_limit_settings(scraper_config, profile=rate_limit_profile)
    requested_user_agent = user_agent or resolve_user_agent(scraper_config)
    if session is not None:
        limiter = getattr(session, _SESSION_LIMITER_ATTR, None)
        session_settings = getattr(session, _SESSION_RATE_LIMIT_ATTR, None)
        effective_user_agent = getattr(session, _SESSION_USER_AGENT_ATTR, None)
        if (
            limiter is None
            or effective_user_agent is None
            or session_settings != requested_settings
            or effective_user_agent != requested_user_agent
        ):
            limiter, effective_user_agent = _configure_session(
                session,
                scraper_config=scraper_config,
                rate_limit_profile=rate_limit_profile,
                user_agent=user_agent,
            )
        do_request = session.request
        request_headers = kwargs.get("headers")
        if request_headers is not None:
            kwargs["headers"] = _headers_with_user_agent(request_headers, user_agent=effective_user_agent)
    else:
        effective_user_agent = requested_user_agent
        limiter = get_shared_rate_limiter(requested_settings)
        kwargs["headers"] = _headers_with_user_agent(kwargs.get("headers"), user_agent=effective_user_agent)
        do_request = requests.request

    for attempt in range(1, attempts + 1):
        try:
            limiter.acquire()
            response = do_request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:  # pragma: no cover - exercised by integration paths
            last_exc = exc
            if attempt == attempts:
                break

            delay = base_delay * (2 ** (attempt - 1))
            LOGGER.warning(
                "HTTP retry %s/%s for %s %s after error: %s",
                attempt,
                attempts,
                method,
                url,
                exc,
            )
            time.sleep(delay)

    raise last_exc  # type: ignore[misc]  # guaranteed non-None by loop logic


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
