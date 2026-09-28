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


# Live arXiv OAI-PMH arXivRaw records (oaipmh.arxiv.org, fetched 2026-09-28), abstracts trimmed.
# 2609.12871 carries comments/journal-ref/doi; 2609.22706 is a bare single-version record.
OAI_RECORD_2609_12871 = """<record>
                <header>
        <identifier>oai:arXiv.org:2609.12871</identifier>
        <datestamp>2026-09-14</datestamp>
            <setSpec>cs:cs:RO</setSpec>
            <setSpec>cs:cs:AI</setSpec>
            <setSpec>cs:cs:CV</setSpec>
            <setSpec>cs:cs:LG</setSpec>
    </header>
            <metadata>
                        <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <id>2609.12871</id>
        <submitter>Philipp Berthold</submitter>
            <version version="v1">
                <date>Fri, 11 Sep 2026 13:55:30 GMT</date>
                <size>6336kb</size>
            </version>
        <title>A Multi-Vehicle Dataset with Camera, LiDAR, and Radar Sensors and Scanned 3D Models for Custom Auto-Annotation using RTK-GNSS</title>
        <authors>Philipp Berthold, Bianca Forkel, Mirko Maehlisch</authors>
        <categories>cs.RO cs.AI cs.CV cs.LG</categories>
            <comments>Paper accompanying the dataset &#34;7V-Scanario&#34;</comments>
            <journal-ref>2025 IEEE Sensor Data Fusion: Trends, Solutions, Applications (SDF)</journal-ref>
            <doi>10.1109/SDF67080.2025.11331266</doi>
            <license>http://arxiv.org/licenses/nonexclusive-distrib/1.0/</license>
            <abstract>Datasets are a crucial element in the development of perception algorithms.</abstract>
    </arXivRaw>
            </metadata>
        </record>"""

OAI_RECORD_2609_22706 = """<record>
                <header>
        <identifier>oai:arXiv.org:2609.22706</identifier>
        <datestamp>2026-09-22</datestamp>
            <setSpec>cs:cs:CV</setSpec>
            <setSpec>cs:cs:IR</setSpec>
    </header>
            <metadata>
                        <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <id>2609.22706</id>
        <submitter>Hao Wang</submitter>
            <version version="v1">
                <date>Sat, 19 Sep 2026 02:39:43 GMT</date>
                <size>5823kb</size>
            </version>
        <title>DOA-SORT: Directional Occlusion-Aware Multi-Object Tracking with Distributional Observations</title>
        <authors>Hao Wang</authors>
        <categories>cs.CV cs.IR</categories>
            <license>http://arxiv.org/licenses/nonexclusive-distrib/1.0/</license>
            <abstract>Identity association in multi-object tracking (MOT) is vulnerable to partial occlusion, truncated detections, and fluctuating confidence scores.</abstract>
    </arXivRaw>
            </metadata>
        </record>"""


def oai_response(verb: str, *records: str, token: str | None = None) -> bytes:
    """An OAI-PMH envelope around ``records``: a GetRecord reply or one ListRecords page."""
    resumption = "" if token is None else f"<resumptionToken>{token}</resumptionToken>"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
        f"<{verb}>{''.join(records)}{resumption}</{verb}></OAI-PMH>"
    ).encode()
