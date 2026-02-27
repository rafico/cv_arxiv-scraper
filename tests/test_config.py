"""Tests for config validation at startup."""

from __future__ import annotations

import copy
import unittest

from app import _validate_config
from tests.helpers import TEST_SCRAPER_CONFIG


class ConfigValidationTests(unittest.TestCase):
    def _valid_config(self) -> dict:
        return copy.deepcopy(TEST_SCRAPER_CONFIG)

    def test_valid_config_passes(self):
        _validate_config(self._valid_config())

    def test_missing_scraper_section(self):
        cfg = self._valid_config()
        del cfg["scraper"]
        with self.assertRaises(ValueError, msg="scraper"):
            _validate_config(cfg)

    def test_missing_whitelists_section(self):
        cfg = self._valid_config()
        del cfg["whitelists"]
        with self.assertRaises(ValueError, msg="whitelists"):
            _validate_config(cfg)

    def test_missing_feed_url(self):
        cfg = self._valid_config()
        del cfg["scraper"]["feed_url"]
        with self.assertRaises(ValueError, msg="feed_url"):
            _validate_config(cfg)

    def test_empty_feed_url(self):
        cfg = self._valid_config()
        cfg["scraper"]["feed_url"] = "   "
        with self.assertRaises(ValueError, msg="non-empty"):
            _validate_config(cfg)

    def test_feed_url_not_string(self):
        cfg = self._valid_config()
        cfg["scraper"]["feed_url"] = 123
        with self.assertRaises(ValueError):
            _validate_config(cfg)

    def test_scraper_not_dict(self):
        cfg = self._valid_config()
        cfg["scraper"] = "bad"
        with self.assertRaises(ValueError, msg="dict"):
            _validate_config(cfg)

    def test_whitelists_not_dict(self):
        cfg = self._valid_config()
        cfg["whitelists"] = ["bad"]
        with self.assertRaises(ValueError, msg="dict"):
            _validate_config(cfg)

    def test_missing_whitelist_key(self):
        for key in ("titles", "authors", "affiliations"):
            cfg = self._valid_config()
            del cfg["whitelists"][key]
            with self.assertRaises(ValueError, msg=key):
                _validate_config(cfg)

    def test_whitelist_not_list(self):
        cfg = self._valid_config()
        cfg["whitelists"]["titles"] = "not a list"
        with self.assertRaises(ValueError, msg="list of strings"):
            _validate_config(cfg)

    def test_whitelist_contains_non_string(self):
        cfg = self._valid_config()
        cfg["whitelists"]["authors"] = ["Alice", 42]
        with self.assertRaises(ValueError, msg="list of strings"):
            _validate_config(cfg)

    def test_config_not_dict(self):
        with self.assertRaises(ValueError, msg="dict"):
            _validate_config("not a dict")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
