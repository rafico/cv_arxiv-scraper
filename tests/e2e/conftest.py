from __future__ import annotations

import copy
import tempfile
import threading
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
import yaml
from werkzeug.serving import make_server

from app import create_app
from app.models import Paper
from app.models import db as _db
from tests.helpers import TEST_SCRAPER_CONFIG


def _make_paper(idx: int, **overrides) -> Paper:
    today = date.today()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    defaults = dict(
        arxiv_id=f"2604.{9000 + idx:04d}",
        title=f"E2E Test Paper {idx}",
        authors=f"Author {idx}, Coauthor {idx}",
        link=f"https://arxiv.org/abs/2604.{9000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2604.{9000 + idx:04d}",
        abstract_text=f"Abstract for paper {idx}.",
        summary_text=f"Summary for paper {idx}.",
        topic_tags=["Vision", "Segmentation"],
        categories=["cs.CV"],
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=50.0 - idx,
        feedback_score=0,
        is_hidden=False,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=now,
    )
    defaults.update(overrides)
    return Paper(**defaults)


@pytest.fixture(scope="session")
def live_server():
    """Start a real Flask HTTP server in a background thread."""
    tmpdir = tempfile.mkdtemp(prefix="e2e_")
    root = Path(tmpdir)
    test_config = copy.deepcopy(TEST_SCRAPER_CONFIG)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(test_config), encoding="utf-8")

    app = create_app(
        {
            "TESTING": True,
            "CONFIG_PATH": str(config_path),
            "SCRAPER_CONFIG": test_config,
            "INSTANCE_PATH": str(root / "instance"),
            "LLM_KEY_PATH": str(root / ".llm_api_key"),
        }
    )

    with app.app_context():
        _db.create_all()

    server = make_server("127.0.0.1", 0, app)
    port = server.socket.getsockname()[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield {"app": app, "url": f"http://127.0.0.1:{port}"}

    server.shutdown()


@pytest.fixture()
def seeded_db(live_server):
    """Seed test papers into the DB, clean up after each test."""
    app = live_server["app"]
    with app.app_context():
        for i in range(3):
            _db.session.add(_make_paper(i))
        _db.session.commit()
        yield
        _db.session.remove()
        for table in reversed(_db.metadata.sorted_tables):
            _db.session.execute(table.delete())
        _db.session.commit()


@pytest.fixture()
def e2e_page(seeded_db, live_server, page):
    """Playwright page pre-configured with the live server base URL."""
    base_url = live_server["url"]
    page.goto(f"{base_url}/?timeframe=all")
    page.wait_for_load_state("networkidle")
    yield page, base_url
