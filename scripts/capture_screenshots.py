#!/usr/bin/env python
"""Regenerate the help/README screenshots against a seeded demo database.

Launches the real Flask app on a background thread with a temporary instance
(so the developer's DB is untouched), seeds a small but rich set of demo
papers/collections/feedback, drives Playwright (Chromium) to capture the UI,
and writes PNG + WebP pairs into app/static/help/.

Usage:
    python scripts/capture_screenshots.py [--headed]

Requires: playwright + a chromium install (`playwright install chromium`).
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import tempfile
import threading
from datetime import date, datetime, timezone
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from werkzeug.serving import make_server

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
STATIC_HELP = REPO_ROOT / "app" / "static" / "help"
THUMBS_DIR = REPO_ROOT / "app" / "static" / "thumbnails"

VIEWPORT = {"width": 1440, "height": 900}


def _demo_config() -> dict:
    """A validated config (reuses the test config) with richer interests so the
    onboarding wizard reads as mostly complete in the captures."""
    from tests.helpers import TEST_SCRAPER_CONFIG

    config = copy.deepcopy(TEST_SCRAPER_CONFIG)
    config["whitelists"] = {
        "authors": ["Jane Doe", "Kaiming He"],
        "affiliations": ["MIT", "Google DeepMind"],
        "titles": ["diffusion", "3D reconstruction", "self-supervised"],
    }
    return config


# (arxiv_id suffix, title, authors, venue, status, cites, stars, why_matched, datasets)
DEMO_PAPERS = [
    (
        "01",
        "Diffusion Models for High-Fidelity 3D Scene Reconstruction",
        "Jane Doe, Kaiming He, Lei Zhang",
        "CVPR",
        2026,
        "oral",
        412,
        1840,
        "Matches your interest in diffusion and 3D reconstruction.",
        ["ScanNet", "KITTI"],
        "Author + Title",
    ),
    (
        "02",
        "Self-Supervised Depth Estimation without Camera Poses",
        "Sofia Müller, Alex Park",
        "ICCV",
        2025,
        "highlight",
        96,
        540,
        "Builds on self-supervised methods you frequently save.",
        ["NYUv2"],
        "Title",
    ),
    (
        "03",
        "NeRF Compression via Learned Octree Pruning",
        "Wei Kim, Diego Park",
        None,
        None,
        None,
        31,
        88,
        None,
        ["DTU"],
        "Title",
    ),
    (
        "04",
        "Open-Vocabulary Segmentation with Vision-Language Priors",
        "Maria Santos, Kaiming He",
        "NeurIPS",
        2025,
        "accepted",
        203,
        1210,
        "Author you follow; strong language-vision grounding.",
        ["COCO", "ADE20K"],
        "Author + Title",
    ),
    (
        "05",
        "Efficient Video Transformers for Real-Time Tracking",
        "Liu Yang, Jane Doe",
        "WACV",
        2026,
        "accepted",
        12,
        0,
        None,
        ["GOT-10k"],
        "Author",
    ),
    (
        "06",
        "Geometry-Aware Diffusion for Novel View Synthesis",
        "Priya Nair, Tom Becker",
        "ECCV",
        2024,
        "accepted",
        158,
        720,
        "Diffusion + novel-view synthesis, right in your wheelhouse.",
        ["RealEstate10K"],
        "Title",
    ),
]


def _placeholder_image(path: Path, label: str, w: int = 320, h: int = 414) -> None:
    """Write a deterministic gradient placeholder so thumbnail routes serve a
    real image without any network/PDF rendering."""
    img = Image.new("RGB", (w, h), "#ffffff")
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        r = int(238 + (79 - 238) * t)
        g = int(242 + (70 - 242) * t)
        b = int(255 + (229 - 255) * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    draw.rectangle([14, 14, w - 14, 70], fill="#ffffff", outline="#e2e8f0")
    draw.text((26, 32), label, fill="#0f172a")
    for i in range(5):
        yy = 110 + i * 26
        draw.line([(26, yy), (w - 26, yy)], fill="#cbd5e1", width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _seed(app) -> list[str]:
    from app.enums import FeedbackAction
    from app.models import (
        Collection,
        Paper,
        PaperCollection,
        PaperFeedback,
        SavedSearch,
        ScrapeRun,
        db,
    )

    today = date.today()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    arxiv_ids: list[str] = []

    with app.app_context():
        db.create_all()
        papers = []
        for sfx, title, authors, venue, vyear, status, cites, stars, why, datasets, mtype in DEMO_PAPERS:
            arxiv_id = f"2604.{sfx}210"
            arxiv_ids.append(arxiv_id)
            insights = {}
            if why:
                insights["why_matched"] = why
            if datasets:
                insights["datasets"] = datasets
            p = Paper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                link=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_link=f"https://arxiv.org/pdf/{arxiv_id}",
                abstract_text=(
                    f"{title}. We present a method that advances the state of the art "
                    "with a simple, scalable design and strong empirical results."
                ),
                summary_text=(
                    "A concise, skimmable summary of the contribution, the key idea, and "
                    "what makes it worth a closer look during triage."
                ),
                topic_tags=["Vision", "Generative", "3D"],
                categories=["cs.CV", "cs.LG"],
                match_type=mtype,
                matched_terms=[t for t in ("diffusion", "3D reconstruction") if t.split()[0].lower() in title.lower()]
                or ["Vision"],
                paper_score=80.0 - len(papers) * 4,
                feedback_score=0,
                is_hidden=False,
                llm_relevance_score=min(10, 6 + len(papers) % 4),
                llm_insights=insights,
                citation_count=cites,
                citation_source="semantic_scholar",
                citation_provenance={"source": "semantic_scholar", "updated_at": now.isoformat()},
                github_repo=("example/repo-" + sfx) if stars else None,
                github_stars=stars or None,
                github_license="MIT" if stars else None,
                arxiv_comment=(f"Accepted to {venue} {vyear}" if venue else None),
                venue=venue,
                venue_year=vyear,
                acceptance_status=status,
                publication_date=today.isoformat(),
                publication_dt=today,
                scraped_date=today.isoformat(),
                scraped_at=now,
            )
            papers.append(p)
            db.session.add(p)
        db.session.commit()

        # A couple of saves + tags/notes to make the UI feel lived-in.
        papers[0].user_tags = ["must-read", "reproduce"]
        papers[0].user_notes = "Compare against our baseline; ask about training cost."
        papers[0].reading_status = "to_read"
        db.session.add(PaperFeedback(paper_id=papers[0].id, action=FeedbackAction.SAVE.value))
        db.session.add(PaperFeedback(paper_id=papers[3].id, action=FeedbackAction.SAVE.value))
        db.session.add(PaperFeedback(paper_id=papers[0].id, action=FeedbackAction.PRIORITY.value))

        collection = Collection(name="Diffusion 3D")
        db.session.add(collection)
        db.session.flush()
        db.session.add(PaperCollection(paper_id=papers[0].id, collection_id=collection.id))
        db.session.add(SavedSearch(name="Oral papers", filters={"q": "diffusion", "timeframe": "all"}))
        db.session.add(ScrapeRun(status="success", started_at=now, finished_at=now))
        db.session.commit()

    return arxiv_ids


def _capture(base_url: str, headed: bool) -> None:
    from playwright.sync_api import sync_playwright

    STATIC_HELP.mkdir(parents=True, exist_ok=True)

    def save(page, name: str, *, full: bool = False, clip=None, selector: str | None = None) -> None:
        png = STATIC_HELP / f"{name}.png"
        if selector is not None:
            page.locator(selector).first.screenshot(path=str(png))
        else:
            page.screenshot(path=str(png), full_page=full, clip=clip)
        Image.open(png).convert("RGB").save(STATIC_HELP / f"{name}.webp", "WEBP", quality=82)
        print("wrote", png.name, "+ .webp")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = ctx.new_page()

        # Dashboard (list density) — README hero + help/start.
        page.goto(f"{base_url}/?timeframe=all", wait_until="networkidle")
        page.wait_for_timeout(400)
        save(page, "papers_dashboard", full=True)

        # Top bar region (Run Scrape) — help/start.
        save(page, "navbar_runscrape", selector="header")

        # First row expanded (TL;DR + details) — help/ui.
        page.locator(".paper-card").first.locator(".card-toggle").click()
        page.wait_for_timeout(250)
        save(page, "dashboard_summary", selector=".paper-card")

        # Visual density grid — help/ui (paper cards).
        page.goto(f"{base_url}/?timeframe=all&density=visual", wait_until="networkidle")
        page.wait_for_timeout(400)
        save(page, "paper_cards", full=True)

        # Settings research setup.
        page.goto(f"{base_url}/settings?section=interests", wait_until="networkidle")
        page.wait_for_timeout(300)
        save(page, "settings_research", full=True)
        save(page, "settings_form", selector="main")

        # Settings ranking controls (summary lines slider lives here).
        page.goto(f"{base_url}/settings?section=controls", wait_until="networkidle")
        page.wait_for_timeout(300)
        save(page, "settings_summary_lines", selector="main")

        # Help landing.
        page.goto(f"{base_url}/help/start", wait_until="networkidle")
        page.wait_for_timeout(300)
        save(page, "help_start", full=True)

        ctx.close()
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headed", action="store_true", help="Run a visible browser (debugging).")
    args = parser.parse_args()

    from app import create_app

    config = _demo_config()
    tmp = Path(tempfile.mkdtemp(prefix="screenshots_"))
    (tmp / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    app = create_app(
        {
            "TESTING": True,
            "CONFIG_PATH": str(tmp / "config.yaml"),
            "SCRAPER_CONFIG": config,
            "INSTANCE_PATH": str(tmp / "instance"),
            "LLM_KEY_PATH": str(tmp / ".llm_api_key"),
        }
    )

    arxiv_ids = _seed(app)

    # Pre-render placeholder thumbnails/teasers so image routes never hit the network.
    written: list[Path] = []
    for arxiv_id in arxiv_ids:
        for suffix in ("", "_teaser"):
            path = THUMBS_DIR / f"{arxiv_id}{suffix}.png"
            _placeholder_image(path, arxiv_id)
            written.append(path)

    server = make_server("127.0.0.1", 0, app)
    port = server.socket.getsockname()[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _capture(f"http://127.0.0.1:{port}", args.headed)
    finally:
        server.shutdown()
        for path in written:
            path.unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)

    print("Done. Screenshots written to", STATIC_HELP)


if __name__ == "__main__":
    main()
