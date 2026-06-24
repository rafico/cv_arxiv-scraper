"""QA round 4 — rate limiter config robustness.

- G14: a YAML-infinity burst (`.inf` -> float('inf')) raises OverflowError inside
  _positive_int's int() call, which is NOT caught, propagating out of
  resolve_rate_limit_settings and breaking ALL HTTP fetching. An infinite burst
  must fall back to the default instead.
- G15: resolve_rate_limit_settings guards that `ingest` is a Mapping but then
  calls rate_limit.get(...) without checking that the nested `rate_limit` is a
  Mapping. A scalar (`fast`), None, or list value raises AttributeError. These
  must fall back to defaults.
"""

from __future__ import annotations

import math
import unittest

from app.services.rate_limiter import (
    DEFAULT_INTERACTIVE_BURST,
    DEFAULT_INTERACTIVE_REQUESTS_PER_SECOND,
    resolve_rate_limit_settings,
)


class RateLimiterConfigRobustnessTests(unittest.TestCase):
    def test_g14_infinite_burst_string_falls_back_to_default(self):
        config = {"ingest": {"rate_limit": {"burst": ".inf"}}}
        settings = resolve_rate_limit_settings(config, profile="interactive")
        self.assertEqual(settings.burst, DEFAULT_INTERACTIVE_BURST)

    def test_g14_infinite_burst_float_falls_back_to_default(self):
        config = {"ingest": {"rate_limit": {"burst": float("inf")}}}
        settings = resolve_rate_limit_settings(config, profile="interactive")
        self.assertEqual(settings.burst, DEFAULT_INTERACTIVE_BURST)

    def test_g14_nan_burst_falls_back_to_default(self):
        config = {"ingest": {"rate_limit": {"burst": math.nan}}}
        settings = resolve_rate_limit_settings(config, profile="interactive")
        self.assertEqual(settings.burst, DEFAULT_INTERACTIVE_BURST)

    def test_g15_scalar_rate_limit_returns_defaults(self):
        config = {"ingest": {"rate_limit": "fast"}}
        settings = resolve_rate_limit_settings(config, profile="interactive")
        self.assertEqual(settings.burst, DEFAULT_INTERACTIVE_BURST)
        self.assertEqual(settings.requests_per_second, DEFAULT_INTERACTIVE_REQUESTS_PER_SECOND)

    def test_g15_none_rate_limit_returns_defaults(self):
        config = {"ingest": {"rate_limit": None}}
        settings = resolve_rate_limit_settings(config, profile="interactive")
        self.assertEqual(settings.burst, DEFAULT_INTERACTIVE_BURST)
        self.assertEqual(settings.requests_per_second, DEFAULT_INTERACTIVE_REQUESTS_PER_SECOND)

    def test_g15_list_rate_limit_returns_defaults(self):
        config = {"ingest": {"rate_limit": [1, 2]}}
        settings = resolve_rate_limit_settings(config, profile="interactive")
        self.assertEqual(settings.burst, DEFAULT_INTERACTIVE_BURST)
        self.assertEqual(settings.requests_per_second, DEFAULT_INTERACTIVE_REQUESTS_PER_SECOND)


if __name__ == "__main__":
    unittest.main()
