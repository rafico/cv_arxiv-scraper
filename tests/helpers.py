from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from app import create_app
from app.constants import DEFAULT_LLM_MODEL
from app.models import db

TEST_SCRAPER_CONFIG = {
    "scraper": {
        "feed_url": "https://example.invalid/rss",
        "rolling_window_days": 0,
        "max_workers": 1,
        "pdf_attempts": 1,
        "pdf_lines_start": 2,
        "pdf_max_header_lines": 50,
        "pdf_smart_header": True,
    },
    "llm": {
        "enabled": False,
        "provider": "openrouter",
        "model": DEFAULT_LLM_MODEL,
        "base_url": "https://openrouter.ai/api/v1",
        "max_concurrent": 4,
    },
    "preferences": {
        "ranking": {
            "author_weight": 44.0,
            "affiliation_weight": 26.0,
            "title_weight": 14.0,
            "ai_weight": 5.0,
            "freshness_half_life_days": 14.0,
        },
        "muted": {
            "authors": [],
            "affiliations": [],
            "topics": [],
        },
    },
    "whitelists": {
        "titles": ["Vision"],
        "affiliations": ["MIT"],
        "authors": ["Jane Doe"],
    },
}


class FlaskDBTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name)
        test_config = copy.deepcopy(TEST_SCRAPER_CONFIG)

        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(test_config), encoding="utf-8")

        db_path = root / "test.db"
        self.app = create_app(
            {
                "TESTING": True,
                "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_path}",
                "CONFIG_PATH": str(config_path),
                "SCRAPER_CONFIG": test_config,
                "LLM_KEY_PATH": str(root / ".llm_api_key"),
            }
        )

        self.ctx = self.app.app_context()
        self.ctx.push()
        db.drop_all()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.ctx.pop()
        self._tmpdir.cleanup()


class DefaultConfigFlaskDBTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.default_config = copy.deepcopy(TEST_SCRAPER_CONFIG)
        (self.root / "config.example.yaml").write_text(yaml.safe_dump(self.default_config), encoding="utf-8")

        db_path = self.root / "test.db"
        self._original_cwd = Path.cwd()
        os.chdir(self.root)
        self.app = create_app(
            {
                "TESTING": True,
                "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_path}",
                "INSTANCE_PATH": str(self.root / "instance"),
                "LLM_KEY_PATH": str(self.root / ".llm_api_key"),
            }
        )
        self.config_path = Path(self.app.config["CONFIG_PATH"])

        self.ctx = self.app.app_context()
        self.ctx.push()
        db.drop_all()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.ctx.pop()
        os.chdir(self._original_cwd)
        self._tmpdir.cleanup()
