"""Wave-2 integration smoke tests: boot, page renders, and cross-feature seams.

Covers the seams between the four Wave-2 features (figures, learned ranker,
paper chat, digest v2) rather than each feature's internals — those live in
their own test modules.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from flask import render_template

from app.models import Paper, PaperFeedback, db
from app.services.email_digest import build_digest_preview, make_one_tap_token
from tests.helpers import FlaskDBTestCase


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        title="Wave2 Smoke Paper",
        authors="Alice, Bob",
        link="https://arxiv.org/abs/2607.00042",
        pdf_link="https://arxiv.org/pdf/2607.00042",
        abstract_text="An abstract about segmentation.",
        summary_text="A summary.",
        topic_tags=["vision"],
        categories=["cs.CV"],
        resource_links=[],
        match_type="Author",
        matched_terms=["Alice"],
        paper_score=42.5,
        feedback_score=0,
        is_hidden=False,
        publication_date="2026-07-01",
        scraped_date="2026-07-01",
        publication_dt=date(2026, 7, 1),
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    defaults.update(overrides)
    return Paper(**defaults)


class Wave2SmokeTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        self.paper = _make_paper()
        db.session.add(self.paper)
        db.session.commit()
        self.client.get("/")
        with self.client.session_transaction() as session:
            self.csrf_token = session["settings_csrf_token"]

    def test_core_pages_render(self):
        for path in ("/", "/discover", "/settings"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)

    def test_dashboard_details_include_chat_panel_without_figures(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("paper-chat-panel", html)
        # No figures cached for this paper, so the strip must render nothing.
        self.assertNotIn("paper-figures", html.replace("paper-figures-", ""))

    def test_details_partial_renders_figures_strip_and_chat_together(self):
        # Seam (c): both Wave-2 includes coexist in _paper_details.html.
        # Annotate exactly the way the dashboard does before it renders cards
        # (rank_score_value, score_breakdown, follow_recommendation, ...), then
        # override figure_indices, which that helper derives from disk.
        from app.routes.dashboard import _enrich_cards_with_feedback_and_related
        from app.services.preferences import get_preferences

        _enrich_cards_with_feedback_and_related([self.paper], [], self.app.config["SCRAPER_CONFIG"])
        self.paper.figure_indices = [1, 2]
        with self.app.test_request_context("/"):
            html = render_template(
                "partials/_paper_details.html",
                paper=self.paper,
                shell="list",
                mendeley_connected=False,
                preferences=get_preferences(self.app),
            )
        self.assertIn("paper-figures", html)
        self.assertIn(f"/papers/{self.paper.id}/figures/1.png", html)
        self.assertIn("paper-chat-panel", html)

    def test_paper_chat_degrades_without_sections_or_llm(self):
        # Paper has no PaperSection rows and no LLM key: expect degraded JSON, not 500.
        response = self.client.post(
            f"/api/papers/{self.paper.id}/chat",
            json={"question": "What datasets are used?"},
            headers={"X-CSRF-Token": self.csrf_token},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["degraded"])
        self.assertFalse(payload["llm_used"])
        self.assertIsNone(payload["answer"])
        self.assertTrue(payload["abstract_only"])

    def test_one_tap_save_creates_feedback_row(self):
        # Seam (b): one-tap -> apply_feedback_action -> learned retrain hook,
        # with a cold learned model (no feedback history, no artifact).
        token = make_one_tap_token(self.app, self.paper.id, "save")
        response = self.client.get(f"/api/feedback/one-tap?token={token}")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("<html", html.lower())
        self.assertIn("Saved", html)
        row = PaperFeedback.query.filter_by(paper_id=self.paper.id, action="save").first()
        self.assertIsNotNone(row)

    def test_digest_preview_renders_new_layout_with_one_tap_links(self):
        # Seam (a): digest calls figure helpers + one-tap token builder offline.
        preview = build_digest_preview(self.app)
        self.assertEqual(preview["papers_count"], 1)
        self.assertFalse(preview["catch_up"])
        self.assertEqual(preview["alerts"], [])
        self.assertIn("Wave2 Smoke Paper", preview["html"])
        self.assertIn("/api/feedback/one-tap?token=", preview["html"])
