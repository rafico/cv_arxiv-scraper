from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from flask import Flask

from tests.helpers import TEST_SCRAPER_CONFIG


class WsgiEntrypointTests(unittest.TestCase):
    def test_wsgi_module_exposes_flask_app(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(TEST_SCRAPER_CONFIG), encoding="utf-8")

            with patch.dict(os.environ, {"CV_ARXIV_CONFIG": str(config_path)}, clear=False):
                sys.modules.pop("wsgi", None)
                module = importlib.import_module("wsgi")
                self.assertIsInstance(module.app, Flask)
                self.assertEqual(Path(module.app.config["CONFIG_PATH"]), config_path.resolve())
                sys.modules.pop("wsgi", None)
