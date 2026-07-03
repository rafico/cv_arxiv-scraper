"""Tests for grounded per-paper chat (app.services.paper_chat + endpoint)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from unittest.mock import patch

import numpy as np

# Importing the module attaches its routes to the shared ``api_bp`` blueprint so the
# endpoint is registered when ``create_app`` runs (mirrors tests/test_rag.py).
import app.routes.api.chat  # noqa: F401  (import-for-side-effect: route registration)
from app.models import Paper, PaperSection, db
from app.services import paper_chat
from tests.helpers import FlaskDBTestCase


def _make_paper(idx: int, **overrides) -> Paper:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = date.today()
    defaults = dict(
        arxiv_id=f"2606.{3000 + idx:04d}",
        title=f"Chat Paper {idx}",
        authors="Author A, Author B",
        link=f"https://arxiv.org/abs/2606.{3000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2606.{3000 + idx:04d}",
        abstract_text=f"Abstract for paper {idx} about vision transformers.",
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


def _add_section(paper: Paper, section_type: str, text: str, order_index: int) -> None:
    db.session.add(PaperSection(paper_id=paper.id, section_type=section_type, text=text, order_index=order_index))


class FakeEmbeddingService:
    """2-dim embeddings: texts containing 'magic' align with a 'magic' question."""

    def encode(self, texts):
        vecs = []
        for text in texts:
            if "magic" in text.lower():
                vecs.append(np.array([1.0, 0.0], dtype=np.float32))
            else:
                vecs.append(np.array([0.0, 1.0], dtype=np.float32))
        return np.asarray(vecs)


class FailingEmbeddingService:
    def encode(self, texts):
        raise RuntimeError("model unavailable")


class FakeClient:
    """LLM stand-in: evidence calls answer from a substring->(score, quote) map."""

    def __init__(self, evidence_scores=None, synthesis="Answer [1]."):
        self.evidence_scores = evidence_scores or {}
        self.synthesis = synthesis
        self.evidence_prompts: list[str] = []
        self.synthesis_prompts: list[str] = []

    def complete_text(self, *, system_prompt, user_prompt, max_tokens, temperature=0.2, **extra):
        if system_prompt == paper_chat._EVIDENCE_SYSTEM_PROMPT:
            self.evidence_prompts.append(user_prompt)
            for needle, (score, quote) in self.evidence_scores.items():
                if needle in user_prompt:
                    return json.dumps({"score": score, "quote": quote})
            return None
        self.synthesis_prompts.append(user_prompt)
        return self.synthesis


def _patch_embeddings(service):
    return patch("app.services.embeddings.get_embedding_service", return_value=service)


def _patch_client(client):
    return patch("app.services.paper_chat._build_client", return_value=client)


class GroundAnswerTests(FlaskDBTestCase):
    def test_strips_unmapped_citations_and_reports_them(self):
        answer, cited, stripped = paper_chat._ground_answer("Uses magic [1]. Also flies [7] and [12].", 2)
        self.assertEqual(answer, "Uses magic [1]. Also flies and .")
        self.assertEqual(cited, [1])
        self.assertEqual(stripped, [7, 12])

    def test_keeps_all_valid_citations(self):
        answer, cited, stripped = paper_chat._ground_answer("A [1] and B [2][2].", 2)
        self.assertEqual(answer, "A [1] and B [2][2].")
        self.assertEqual(cited, [1, 2])
        self.assertEqual(stripped, [])


class RetrievalTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.paper = _make_paper(0)
        self.other = _make_paper(1)
        db.session.add_all([self.paper, self.other])
        db.session.commit()
        _add_section(self.paper, "method", "We use a magic decoder for segmentation.", 1)
        _add_section(self.paper, "conclusion", "Future work remains.", 2)
        _add_section(self.other, "results", "OTHER PAPER results text about magic too.", 1)
        db.session.commit()

    def test_retrieval_restricted_to_the_target_paper(self):
        with _patch_embeddings(FakeEmbeddingService()):
            result = paper_chat.answer_paper_question(self.paper.id, "what magic method?")
        types = {s["section_type"] for s in result["sections"]}
        self.assertEqual(types, {"abstract", "method", "conclusion"})
        self.assertNotIn("results", types)
        for section in result["sections"]:
            self.assertNotIn("OTHER PAPER", section["snippet"])

    def test_ranking_orders_by_similarity_with_scores(self):
        with _patch_embeddings(FakeEmbeddingService()):
            result = paper_chat.answer_paper_question(self.paper.id, "what magic method?")
        self.assertEqual(result["sections"][0]["section_type"], "method")
        self.assertAlmostEqual(result["sections"][0]["score"], 1.0, places=5)
        # No LLM configured in the test config: honest degradation.
        self.assertTrue(result["degraded"])
        self.assertIsNone(result["answer"])
        self.assertFalse(result["llm_error"])
        self.assertFalse(result["abstract_only"])

    def test_abstract_always_survives_the_top_k_cut(self):
        # 9 more high-similarity sections push the (non-matching) abstract past top-8.
        for i in range(9):
            _add_section(self.paper, "results", f"magic filler block {i}", 10 + i)
        db.session.commit()
        with _patch_embeddings(FakeEmbeddingService()):
            result = paper_chat.answer_paper_question(self.paper.id, "what magic method?")
        types = [s["section_type"] for s in result["sections"]]
        self.assertEqual(len(types), paper_chat._TOP_CHUNKS)
        self.assertIn("abstract", types)

    def test_encode_failure_degrades_to_document_order(self):
        with _patch_embeddings(FailingEmbeddingService()):
            result = paper_chat.answer_paper_question(self.paper.id, "anything")
        types = [s["section_type"] for s in result["sections"]]
        self.assertEqual(types, ["abstract", "method", "conclusion"])
        self.assertIsNone(result["sections"][0]["score"])


class EvidenceAndSynthesisTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.paper = _make_paper(0)
        db.session.add(self.paper)
        db.session.commit()
        _add_section(self.paper, "method", "We use a magic decoder for segmentation.", 1)
        _add_section(self.paper, "conclusion", "UNRELATED closing remarks.", 2)
        db.session.commit()

    def test_evidence_scoring_drops_low_scorers(self):
        client = FakeClient(
            evidence_scores={
                "decoder for segmentation": (9, "We use a magic decoder for segmentation."),
                "UNRELATED": (1, ""),
                "Abstract for paper": (2, ""),
            },
            synthesis="It uses a magic decoder [1].",
        )
        with _patch_embeddings(FakeEmbeddingService()), _patch_client(client):
            result = paper_chat.answer_paper_question(self.paper.id, "what magic decoder?")

        self.assertEqual(result["answer"], "It uses a magic decoder [1].")
        self.assertTrue(result["llm_used"])
        self.assertFalse(result["degraded"])
        self.assertFalse(result["evidence_degraded"])
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["section_type"], "method")
        self.assertTrue(result["citations"][0]["cited"])
        # Low scorers never reach the synthesis prompt.
        self.assertNotIn("UNRELATED", client.synthesis_prompts[0])

    def test_unmapped_citations_are_stripped_from_the_answer(self):
        client = FakeClient(
            evidence_scores={"decoder for segmentation": (9, "quote")},
            synthesis="Magic decoder [1]. Fabricated claim [7].",
        )
        with _patch_embeddings(FakeEmbeddingService()), _patch_client(client):
            result = paper_chat.answer_paper_question(self.paper.id, "what magic decoder?")
        self.assertNotIn("[7]", result["answer"])
        self.assertIn("[1]", result["answer"])
        self.assertEqual(result["stripped_citations"], [7])

    def test_evidence_failure_falls_back_to_retrieval_order(self):
        client = FakeClient(evidence_scores={}, synthesis="Best-effort answer [1].")
        with _patch_embeddings(FakeEmbeddingService()), _patch_client(client):
            result = paper_chat.answer_paper_question(self.paper.id, "what magic decoder?")
        self.assertTrue(result["evidence_degraded"])
        self.assertEqual(result["answer"], "Best-effort answer [1].")
        # Fallback keeps retrieval order: the best-matching section leads.
        self.assertEqual(result["citations"][0]["section_type"], "method")
        self.assertEqual(result["citations"][0]["quote"], "")

    def test_all_low_scores_short_circuits_to_not_addressed(self):
        client = FakeClient(
            evidence_scores={
                "decoder for segmentation": (1, ""),
                "UNRELATED": (0, ""),
                "Abstract for paper": (1, ""),
            },
        )
        with _patch_embeddings(FakeEmbeddingService()), _patch_client(client):
            result = paper_chat.answer_paper_question(self.paper.id, "quantum finance?")
        self.assertEqual(result["answer"], paper_chat._NOT_ADDRESSED)
        self.assertTrue(result["llm_used"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(client.synthesis_prompts, [])

    def test_synthesis_failure_sets_llm_error(self):
        client = FakeClient(evidence_scores={"decoder for segmentation": (9, "q")}, synthesis=None)
        with _patch_embeddings(FakeEmbeddingService()), _patch_client(client):
            result = paper_chat.answer_paper_question(self.paper.id, "what magic decoder?")
        self.assertTrue(result["llm_error"])
        self.assertTrue(result["degraded"])
        self.assertIsNone(result["answer"])

    def test_history_reaches_the_synthesis_prompt_sanitized(self):
        client = FakeClient(evidence_scores={"decoder for segmentation": (9, "q")}, synthesis="Follow-up [1].")
        history = [
            {"question": "What task?", "answer": "Segmentation."},
            {"bogus": True},
            "not a dict",
            {"question": "", "answer": "ignored"},
        ]
        with _patch_embeddings(FakeEmbeddingService()), _patch_client(client):
            result = paper_chat.answer_paper_question(self.paper.id, "and the magic decoder?", history)
        self.assertEqual(result["answer"], "Follow-up [1].")
        prompt = client.synthesis_prompts[0]
        self.assertIn("Q: What task?", prompt)
        self.assertIn("A: Segmentation.", prompt)

    def test_abstract_only_paper_answers_from_abstract(self):
        bare = _make_paper(5, abstract_text="Only a magic abstract exists here.")
        db.session.add(bare)
        db.session.commit()
        client = FakeClient(
            evidence_scores={"magic abstract": (8, "Only a magic abstract exists here.")},
            synthesis="From the abstract [1].",
        )
        with _patch_embeddings(FakeEmbeddingService()), _patch_client(client):
            result = paper_chat.answer_paper_question(bare.id, "any magic?")
        self.assertTrue(result["abstract_only"])
        self.assertEqual(result["answer"], "From the abstract [1].")
        self.assertEqual(result["citations"][0]["section_type"], "abstract")


class PaperChatEndpointTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        self.client.get("/")
        with self.client.session_transaction() as session:
            self.csrf_token = session["settings_csrf_token"]
        self.paper = _make_paper(0)
        db.session.add(self.paper)
        db.session.commit()
        _add_section(self.paper, "method", "We use a magic decoder for segmentation.", 1)
        db.session.commit()

    def _post(self, paper_id, payload):
        return self.client.post(
            f"/api/papers/{paper_id}/chat",
            json=payload,
            headers={"X-CSRF-Token": self.csrf_token},
        )

    def test_requires_csrf(self):
        response = self.client.post(f"/api/papers/{self.paper.id}/chat", json={"question": "hi"})
        self.assertEqual(response.status_code, 400)

    def test_missing_question_is_400(self):
        response = self._post(self.paper.id, {"question": "   "})
        self.assertEqual(response.status_code, 400)

    def test_oversized_question_is_400(self):
        response = self._post(self.paper.id, {"question": "x" * 2001})
        self.assertEqual(response.status_code, 400)

    def test_wrong_typed_history_is_400(self):
        response = self._post(self.paper.id, {"question": "hi", "history": "nope"})
        self.assertEqual(response.status_code, 400)

    def test_unknown_paper_is_404(self):
        response = self._post(999999, {"question": "hi"})
        self.assertEqual(response.status_code, 404)

    def test_degraded_no_llm_returns_200_sections(self):
        with _patch_embeddings(FakeEmbeddingService()):
            response = self._post(self.paper.id, {"question": "what magic decoder?"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsNone(data["answer"])
        self.assertTrue(data["degraded"])
        self.assertEqual(data["sections"][0]["section_type"], "method")

    def test_llm_failure_maps_to_502(self):
        client = FakeClient(evidence_scores={"decoder for segmentation": (9, "q")}, synthesis=None)
        with _patch_embeddings(FakeEmbeddingService()), _patch_client(client):
            response = self._post(self.paper.id, {"question": "what magic decoder?"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("error", response.get_json())

    def test_happy_path_returns_answer_and_citations(self):
        client = FakeClient(
            evidence_scores={"decoder for segmentation": (9, "We use a magic decoder for segmentation.")},
            synthesis="It uses a magic decoder [1]. Bogus [9].",
        )
        with _patch_embeddings(FakeEmbeddingService()), _patch_client(client):
            response = self._post(self.paper.id, {"question": "what magic decoder?", "history": []})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["answer"], "It uses a magic decoder [1]. Bogus .")
        self.assertNotIn("[9]", data["answer"])
        self.assertEqual(data["stripped_citations"], [9])
        self.assertEqual(data["citations"][0]["n"], 1)
        self.assertEqual(data["citations"][0]["section_type"], "method")
        self.assertTrue(data["citations"][0]["cited"])
        self.assertFalse(data["abstract_only"])
