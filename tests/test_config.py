"""Tests for config validation at startup."""

from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_llm_enabled_requires_key_source(self):
        cfg = self._valid_config()
        cfg["llm"]["enabled"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with patch.dict(os.environ, {}, clear=False):
                with self.assertRaises(ValueError, msg="API key"):
                    _validate_config(cfg, config_path=config_path)

    def test_llm_enabled_accepts_env_var(self):
        cfg = self._valid_config()
        cfg["llm"]["enabled"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
                _validate_config(cfg, config_path=config_path)

    def test_llm_enabled_accepts_key_file(self):
        cfg = self._valid_config()
        cfg["llm"]["enabled"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".llm_api_key").write_text("file-key", encoding="utf-8")
            _validate_config(cfg, config_path=root / "config.yaml")

    def test_preferences_ranking_values_must_be_positive(self):
        cfg = self._valid_config()
        cfg["preferences"]["ranking"]["author_weight"] = 0
        with self.assertRaises(ValueError, msg="positive"):
            _validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
