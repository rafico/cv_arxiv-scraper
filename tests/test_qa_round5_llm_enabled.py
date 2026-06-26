"""QA round 5 regression test — R5-con7 (S2): a hand-edited scalar ``llm.enabled``
("false"/"no"/"off") must be read through _is_truthy_flag, not raw truthiness.

Raw truthiness treats the *string* "false" as enabled, which (a) crashes startup
when no API key is configured and (b) silently runs the (costly) LLM when the user
meant to disable it. Mirrors the round-3 F19 fix for scheduler.enabled.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
from app import _validate_config
from app.services.scrape_engine import _create_llm_client
from tests.helpers import TEST_SCRAPER_CONFIG, FlaskDBTestCase


class ValidateConfigLlmEnabledTests(unittest.TestCase):
    @patch.object(app_module, "_llm_api_key_available", return_value=False)
    def test_false_string_does_not_require_api_key(self, _mock):
        config = copy.deepcopy(TEST_SCRAPER_CONFIG)
        config["llm"]["enabled"] = "false"
        # "false" is a disabled flag, so no API key is required: must not raise.
        _validate_config(config, config_path=None)

    @patch.object(app_module, "_llm_api_key_available", return_value=False)
    def test_true_bool_still_requires_api_key(self, _mock):
        config = copy.deepcopy(TEST_SCRAPER_CONFIG)
        config["llm"]["enabled"] = True
        with self.assertRaises(ValueError):
            _validate_config(config, config_path=None)


class CreateLlmClientEnabledTests(FlaskDBTestCase):
    @patch("app.services.scrape_engine.LLMClient")
    def test_false_string_disables_client(self, mock_client):
        # A real key is present, so a truthy reading would build a client.
        Path(self.app.config["LLM_KEY_PATH"]).write_text("sk-test-key", encoding="utf-8")
        self.app.config["SCRAPER_CONFIG"]["llm"] = {
            "enabled": "false",
            "provider": "openrouter",
            "model": "m",
            "base_url": "https://openrouter.ai/api/v1",
            "max_concurrent": 4,
        }

        client, _interests = _create_llm_client(self.app)

        self.assertIsNone(client)
        mock_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
