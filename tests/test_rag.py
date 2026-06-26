"""Tests for conversational RAG over the saved corpus (app.services.rag + endpoint)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

# Importing the module attaches its route to the shared ``api_bp`` blueprint so the
# endpoint is registered when ``create_app`` runs. In production this happens via the
# import tuple in ``app/routes/api/__init__.py`` (see REPORT for the wiring snippet).
import app.routes.api.chat  # noqa: E402,F401  (import-for-side-effect: route registration)
from app.enums import FeedbackAction
from app.models import Paper, PaperFeedback, db
from app.services import rag
from tests.helpers import FlaskDBTestCase


def _make_paper(idx: int, **overrides) -> Paper:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    defaults = dict(
        arxiv_id=f"2606.{2000 + idx:04d}",
        title=f"RAG Paper {idx}",
        authors="Author A, Author B",
        link=f"https://arxiv.org/abs/2606.{2000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2606.{2000 + idx:04d}",
        abstract_text=f"Abstract for paper {idx} about vision transformers and segmentation.",
        summary_text=f"Summary {idx}",
        topic_tags=["Segmentation"],
        categories=["cs.CV"],
        match_type="Title",
        matched_terms=["Vision"],
        paper_score=10.0 + idx,
        feedback_score=0,
        is_hidden=False,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=now,
    )
    defaults.update(overrides)
    return Paper(**defaults)


def _save(paper: Paper) -> None:
    db.session.add(PaperFeedback(paper_id=paper.id, action=FeedbackAction.SAVE.value))


class RetrieveSavedContextTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.papers = [_make_paper(i) for i in range(4)]
        db.session.add_all(self.papers)
        db.session.commit()
        # Save the first three; leave the fourth unsaved.
        for paper in self.papers[:3]:
            _save(paper)
        db.session.commit()
        self.saved_ids = {p.id for p in self.papers[:3]}

    def test_filters_hybrid_results_to_saved_papers(self):
        unsaved = self.papers[3]
        ranked = [
            {"paper_id": unsaved.id, "rrf_score": 0.9, "bm25_rank": 1, "semantic_rank": 1},
            {"paper_id": self.papers[0].id, "rrf_score": 0.5, "bm25_rank": 2, "semantic_rank": 2},
            {"paper_id": self.papers[1].id, "rrf_score": 0.3, "bm25_rank": 3, "semantic_rank": 3},
        ]
        with patch("app.services.rag.search_hybrid", return_value=ranked):
            result = rag.retrieve_saved_context("vision transformers", top_k=6)

        returned_ids = {s["paper_id"] for s in result["sources"]}
        self.assertNotIn(unsaved.id, returned_ids)
        self.assertTrue(returned_ids.issubset(self.saved_ids))
        # Highest-ranked saved paper comes first.
        self.assertEqual(result["sources"][0]["paper_id"], self.papers[0].id)
        self.assertIn("Title: RAG Paper 0", result["context"])
        self.assertFalse(result["no_saved_papers"])

    def test_falls_back_to_saved_papers_when_hybrid_empty(self):
        with patch("app.services.rag.search_hybrid", return_value=[]):
            result = rag.retrieve_saved_context("anything", top_k=6)

        returned_ids = {s["paper_id"] for s in result["sources"]}
        self.assertEqual(returned_ids, self.saved_ids)


class AnswerQueryTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.papers = [_make_paper(i) for i in range(3)]
        db.session.add_all(self.papers)
        db.session.commit()
        for paper in self.papers[:2]:
            _save(paper)
        db.session.commit()
        self.ranked = [
            {"paper_id": self.papers[0].id, "rrf_score": 0.5, "bm25_rank": 1, "semantic_rank": 1},
            {"paper_id": self.papers[1].id, "rrf_score": 0.3, "bm25_rank": 2, "semantic_rank": 2},
        ]

    def test_llm_disabled_returns_sources_without_synthesis(self):
        # Config default has llm.enabled=False, so _build_client returns None.
        with patch("app.services.rag.search_hybrid", return_value=self.ranked):
            result = rag.answer_query("what is new in segmentation?", app=self.app)

        self.assertIsNone(result["synthesis"])
        self.assertFalse(result["llm_used"])
        self.assertFalse(result["no_saved_papers"])
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["query"], "what is new in segmentation?")

    def test_llm_enabled_returns_synthesis(self):
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded answer citing RAG Paper 0."))]
        )
        fake_client = SimpleNamespace(_create_completion=lambda **kwargs: fake_response)

        with (
            patch("app.services.rag.search_hybrid", return_value=self.ranked),
            patch("app.services.rag._build_client", return_value=fake_client),
        ):
            result = rag.answer_query("summarize my saved work", app=self.app)

        self.assertEqual(result["synthesis"], "Grounded answer citing RAG Paper 0.")
        self.assertTrue(result["llm_used"])
        self.assertEqual(len(result["sources"]), 2)

    def test_llm_failure_degrades_to_none(self):
        def _boom(**kwargs):
            raise RuntimeError("network down")

        fake_client = SimpleNamespace(_create_completion=_boom)
        with (
            patch("app.services.rag.search_hybrid", return_value=self.ranked),
            patch("app.services.rag._build_client", return_value=fake_client),
        ):
            result = rag.answer_query("anything", app=self.app)

        self.assertIsNone(result["synthesis"])
        self.assertFalse(result["llm_used"])

    def test_no_saved_papers_returns_friendly_empty(self):
        # Remove all save feedback.
        PaperFeedback.query.delete()
        db.session.commit()

        result = rag.answer_query("anything at all", app=self.app)
        self.assertTrue(result["no_saved_papers"])
        self.assertEqual(result["sources"], [])
        self.assertIsNone(result["synthesis"])
        self.assertFalse(result["llm_used"])


class ChatEndpointTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        self.client.get("/")
        with self.client.session_transaction() as session:
            self.csrf_token = session["settings_csrf_token"]
        paper = _make_paper(0)
        db.session.add(paper)
        db.session.commit()
        _save(paper)
        db.session.commit()

    def test_chat_requires_csrf(self):
        response = self.client.post("/api/corpus/chat", json={"query": "hi"})
        self.assertEqual(response.status_code, 400)

    def test_chat_rejects_empty_query(self):
        response = self.client.post(
            "/api/corpus/chat",
            json={"query": "   "},
            headers={"X-CSRF-Token": self.csrf_token},
        )
        self.assertEqual(response.status_code, 400)

    def test_chat_returns_200_json_with_sources(self):
        ranked = [{"paper_id": 1, "rrf_score": 0.5, "bm25_rank": 1, "semantic_rank": 1}]
        with patch("app.services.rag.search_hybrid", return_value=ranked):
            response = self.client.post(
                "/api/corpus/chat",
                json={"query": "what did I save?"},
                headers={"X-CSRF-Token": self.csrf_token},
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertFalse(data["no_saved_papers"])
        self.assertFalse(data["llm_used"])
        self.assertIsNone(data["synthesis"])
        self.assertEqual(len(data["sources"]), 1)

    def test_chat_no_saved_papers_friendly_payload(self):
        PaperFeedback.query.delete()
        db.session.commit()
        response = self.client.post(
            "/api/corpus/chat",
            json={"query": "anything"},
            headers={"X-CSRF-Token": self.csrf_token},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["no_saved_papers"])
        self.assertIn("message", data)
        self.assertEqual(data["sources"], [])
