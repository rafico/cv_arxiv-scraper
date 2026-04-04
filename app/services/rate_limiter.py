"""Shared token-bucket rate limiting helpers."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_INTERACTIVE_REQUESTS_PER_SECOND = 4.0
DEFAULT_INTERACTIVE_BURST = 4
DEFAULT_BULK_REQUESTS_PER_SECOND = 1.0 / 3.0
DEFAULT_BULK_BURST = 1


@dataclass(frozen=True, slots=True)
class RateLimitSettings:
    """Resolved rate-limit settings for a named HTTP profile."""

    profile: str
    requests_per_second: float
    burst: int


class TokenBucketRateLimiter:
    """Thread-safe token bucket limiter."""

    def __init__(
        self,
        *,
        requests_per_second: float,
        burst: int,
        time_fn=time.monotonic,
        sleep_fn=time.sleep,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if burst < 1:
            raise ValueError("burst must be at least 1")

        self.requests_per_second = float(requests_per_second)
        self.burst = int(burst)
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._tokens = float(burst)
        self._last_refill = self._time_fn()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_refill)
        if elapsed > 0:
            self._tokens = min(self.burst, self._tokens + (elapsed * self.requests_per_second))
            self._last_refill = now

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until tokens are available. Returns total wait time."""
        if tokens <= 0:
            return 0.0

        waited = 0.0
        with self._lock:
            while True:
                now = self._time_fn()
                self._refill(now)
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited

                delay = (tokens - self._tokens) / self.requests_per_second
                self._sleep_fn(delay)
                waited += delay


_SHARED_LIMITERS: dict[RateLimitSettings, TokenBucketRateLimiter] = {}
_SHARED_LIMITERS_LOCK = threading.Lock()


def _positive_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
        if float(parsed) != float(value):
            return None
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def resolve_rate_limit_settings(
    scraper_config: Mapping[str, Any] | None,
    *,
    profile: str = "interactive",
) -> RateLimitSettings:
    """Resolve effective rate limits for the requested profile."""
    profile_name = str(profile or "interactive").strip().lower()
    if profile_name not in {"interactive", "bulk"}:
        raise ValueError(f"Unknown rate limit profile: {profile}")

    ingest = scraper_config.get("ingest", {}) if isinstance(scraper_config, Mapping) else {}
    rate_limit = ingest.get("rate_limit", {}) if isinstance(ingest, Mapping) else {}

    configured_rps = _positive_float(rate_limit.get("requests_per_second"))
    configured_burst = _positive_int(rate_limit.get("burst"))

    requests_per_second = configured_rps or DEFAULT_INTERACTIVE_REQUESTS_PER_SECOND
    burst = configured_burst or DEFAULT_INTERACTIVE_BURST

    if profile_name == "bulk":
        requests_per_second = min(requests_per_second, DEFAULT_BULK_REQUESTS_PER_SECOND)
        burst = min(burst, DEFAULT_BULK_BURST)

    return RateLimitSettings(
        profile=profile_name,
        requests_per_second=requests_per_second,
        burst=burst,
    )


def get_shared_rate_limiter(settings: RateLimitSettings) -> TokenBucketRateLimiter:
    """Return a shared limiter for the given settings."""
    with _SHARED_LIMITERS_LOCK:
        limiter = _SHARED_LIMITERS.get(settings)
        if limiter is None:
            limiter = TokenBucketRateLimiter(
                requests_per_second=settings.requests_per_second,
                burst=settings.burst,
            )
            _SHARED_LIMITERS[settings] = limiter
        return limiter
