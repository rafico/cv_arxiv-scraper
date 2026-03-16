"""HTTP helpers with retry/backoff defaults."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


def request_with_backoff(
    method: str,
    url: str,
    *,
    attempts: int = 3,
    base_delay: float = 1.25,
    timeout: int = 30,
    session: requests.Session | None = None,
    **kwargs: Any,
) -> requests.Response:
    """Run an HTTP request with bounded retries and exponential backoff."""
    last_exc: Exception | None = None
    do_request = session.request if session else requests.request

    for attempt in range(1, attempts + 1):
        try:
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


def create_session(pool_size: int = 10) -> requests.Session:
    """Create a session with connection pooling for concurrent downloads."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
