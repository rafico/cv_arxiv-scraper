from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from app import create_app
from app.models import db

TEST_SCRAPER_CONFIG = {
    "scraper": {
        "feed_url": "https://example.invalid/rss",
        "max_workers": 1,
        "pdf_attempts": 1,
        "pdf_lines_start": 2,
        "pdf_max_header_lines": 50,
        "pdf_smart_header": True,
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

        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(TEST_SCRAPER_CONFIG), encoding="utf-8")

        db_path = root / "test.db"
        self.app = create_app(
            {
                "TESTING": True,
                "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_path}",
                "CONFIG_PATH": str(config_path),
                "SCRAPER_CONFIG": TEST_SCRAPER_CONFIG,
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
