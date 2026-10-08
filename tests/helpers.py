from __future__ import annotations

import copy
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import requests
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


class FakeMendeley:
    """An in-memory api.mendeley.com, to patch over ``app.services.mendeley.requests.get``/``.post``.

    Status codes follow the API's own declaration (api.mendeley.com/apidocs): a second
    folder of the same name and a document already in the folder are 409, an unknown
    folder or document is 404, and trashed documents are left out of listings unless
    ``include_trashed=true``. It answers with real ``requests.Response`` objects, so status
    checks, JSON decoding and the Link-header pagination are the library's own.

    ``calls`` records ``(method, path)`` and ``headers`` the headers of each call;
    ``intercept(method, path, body)`` may return an HTTP status to fail a call with, or raise.
    """

    BASE = "https://api.mendeley.com"

    def __init__(self, page_size: int = 500):
        self.page_size = page_size
        self.folders: dict[str, str] = {}  # id -> name
        self.documents: dict[str, dict] = {}  # id -> document
        self.folder_documents: dict[str, list[str]] = {}  # folder id -> document ids
        self.trashed: set[str] = set()  # document ids in the trash
        self.calls: list[tuple[str, str]] = []
        self.headers: list[dict] = []
        self.intercept = lambda method, path, body: None
        self._ids = itertools.count(1)

    def get(self, url, **kwargs):
        return self._handle("get", url, kwargs)

    def post(self, url, **kwargs):
        return self._handle("post", url, kwargs)

    @staticmethod
    def _response(status: int, body, url: str, next_url: str | None = None) -> requests.Response:
        response = requests.Response()
        response.status_code = status
        response.url = url
        response._content = b"" if body is None else json.dumps(body).encode()
        if next_url:
            response.headers["Link"] = f'<{next_url}>; rel="next"'
        return response

    def _handle(self, method: str, url: str, kwargs: dict) -> requests.Response:
        parts = urlsplit(url)
        path, body = parts.path, kwargs.get("json")
        self.calls.append((method, path))
        self.headers.append(kwargs.get("headers") or {})
        status = self.intercept(method, path, body)
        if status:
            return self._response(status, {"message": "intercepted"}, url)

        segments = path.strip("/").split("/")
        in_folder = len(segments) == 3 and segments[0] == "folders" and segments[2] == "documents"
        if method == "get":
            query = {**parse_qs(parts.query), **{k: [str(v)] for k, v in (kwargs.get("params") or {}).items()}}
            hidden = set() if query.get("include_trashed") == ["true"] else self.trashed
            if segments == ["folders"]:
                items = [{"id": folder_id, "name": name} for folder_id, name in self.folders.items()]
            elif segments == ["documents"]:
                items = [document for document in self.documents.values() if document["id"] not in hidden]
            elif in_folder and segments[1] in self.folders:
                items = [{"id": doc_id} for doc_id in self.folder_documents[segments[1]] if doc_id not in hidden]
            else:
                return self._response(404, {"message": "not found"}, url)
            # The marker is opaque to clients: they must follow the Link URL as given.
            limit = min(int(query.get("limit", ["20"])[0]), self.page_size)
            start = int(query.get("marker", ["0"])[0])
            more = start + limit < len(items)
            carried = "&include_trashed=true" if query.get("include_trashed") == ["true"] else ""
            next_url = f"{self.BASE}{path}?limit={limit}&marker={start + limit}{carried}" if more else None
            return self._response(200, items[start : start + limit], url, next_url)
        if segments == ["folders"]:
            if body["name"] in self.folders.values():
                return self._response(409, {"message": "A folder with the same name already exists"}, url)
            folder_id = f"folder-{next(self._ids)}"
            self.folders[folder_id] = body["name"]
            self.folder_documents[folder_id] = []
            return self._response(201, {"id": folder_id, "name": body["name"]}, url)
        if segments == ["documents"]:
            document = {**body, "id": f"doc-{next(self._ids)}"}
            self.documents[document["id"]] = document
            return self._response(201, document, url)
        if in_folder and segments[1] in self.folders and body["id"] in self.documents:
            if body["id"] in self.folder_documents[segments[1]]:
                return self._response(409, {"message": "Document already exists in folder"}, url)
            self.folder_documents[segments[1]].append(body["id"])
            return self._response(201, None, url)
        return self._response(404, {"message": "Folder or document not found"}, url)
