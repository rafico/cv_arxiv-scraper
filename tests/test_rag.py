"""Tests for conversational RAG over a collection / the saved papers / the corpus (app.services.rag + endpoint)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

# Importing the module attaches its route to the shared ``api_bp`` blueprint so the
# endpoint is registered when ``create_app`` runs. In production this happens via the
# import tuple in ``app/routes/api/__init__.py`` (see REPORT for the wiring snippet).
import app.routes.api.chat  # noqa: E402,F401  (import-for-side-effect: route registration)
from app.enums import FeedbackAction
from app.models import Collection, Paper, PaperCollection, PaperFeedback, PaperSection, db
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


class FakeEmbeddingService:
    """2-dim embeddings: 'voxel' texts align with a 'voxel' query; paper vectors are given per id."""

    def __init__(self, paper_vectors=None):
        self.paper_vectors = paper_vectors or {}

    def encode(self, texts):
        return np.asarray([[1.0, 0.0] if "voxel" in t.lower() else [0.0, 1.0] for t in texts], dtype=np.float32)

    def get_paper_vectors(self, paper_ids):
        found = [pid for pid in paper_ids if pid in self.paper_vectors]
        return found, np.asarray([self.paper_vectors[pid] for pid in found], dtype=np.float32)


def _patch_embeddings(service):
    return patch("app.services.embeddings.get_embedding_service", return_value=service)


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

    def test_saved_scope_ranks_saved_papers_exactly(self):
        unsaved = self.papers[3]
        vectors = {p.id: [0.0, 1.0] for p in self.papers}
        vectors[self.papers[2].id] = [1.0, 0.0]  # best match for a "voxel" query
        with _patch_embeddings(FakeEmbeddingService(vectors)):
            result = rag.retrieve_saved_context("voxel occupancy", top_k=6)

        returned_ids = [s["paper_id"] for s in result["sources"]]
        self.assertEqual(result["scope"], "saved")
        self.assertNotIn(unsaved.id, returned_ids)
        self.assertEqual(set(returned_ids), self.saved_ids)
        self.assertEqual(returned_ids[0], self.papers[2].id)
        self.assertEqual([s["n"] for s in result["sources"]], [1, 2, 3])
        self.assertIn("[1] RAG Paper 2", result["context"])

    def test_unembedded_saved_papers_still_returned(self):
        # Replaces the old "hybrid empty -> saved ids" fallback: an empty index must not hide saves.
        with patch("app.services.rag.search_hybrid", return_value=[]):
            result = rag.retrieve_saved_context("anything", top_k=6)

        self.assertEqual({s["paper_id"] for s in result["sources"]}, self.saved_ids)

    def test_scoped_member_outside_hybrid_top_k_is_returned(self):
        member = self.papers[3]  # unsaved, and missing from the global hybrid hits
        vectors = {p.id: [0.0, 1.0] for p in self.papers}
        vectors[member.id] = [1.0, 0.0]
        hits = [{"paper_id": self.papers[0].id, "rrf_score": 0.9}]
        with (
            patch("app.services.rag.search_hybrid", return_value=hits),
            _patch_embeddings(FakeEmbeddingService(vectors)),
        ):
            result = rag.retrieve_saved_context(
                "voxel occupancy", top_k=1, paper_ids=[self.papers[0].id, self.papers[1].id, member.id]
            )

        self.assertEqual(result["scope"], "collection")
        self.assertEqual([s["paper_id"] for s in result["sources"]], [member.id])

    def test_no_saves_searches_whole_corpus(self):
        PaperFeedback.query.delete()
        db.session.commit()
        unsaved = self.papers[3]
        hits = [{"paper_id": unsaved.id, "rrf_score": 0.5}]
        with patch("app.services.rag.search_hybrid", return_value=hits):
            result = rag.retrieve_saved_context("anything", top_k=6)

        self.assertEqual(result["scope"], "corpus")
        self.assertEqual([s["paper_id"] for s in result["sources"]], [unsaved.id])

    def test_corpus_scope_skips_hidden_papers(self):
        PaperFeedback.query.delete()
        skipped = self.papers[3]
        skipped.is_hidden = True  # what a skip sets
        db.session.commit()
        hits = [{"paper_id": skipped.id, "rrf_score": 0.9}, {"paper_id": self.papers[0].id, "rrf_score": 0.5}]
        with patch("app.services.rag.search_hybrid", return_value=hits):
            result = rag.retrieve_saved_context("anything", top_k=1)

        self.assertEqual([s["paper_id"] for s in result["sources"]], [self.papers[0].id])

    def test_excerpt_drops_numeric_in_text_citations(self):
        # A body "[3]" would read as source [3] and survive grounding.
        paper = self.papers[0]
        db.session.add(
            PaperSection(
                paper_id=paper.id, section_type="method", text="We follow DeiT [3] and [4, 5] here.", order_index=1
            )
        )
        db.session.commit()
        with _patch_embeddings(FakeEmbeddingService()):
            result = rag.retrieve_saved_context("distillation", top_k=6, paper_ids=[paper.id])

        self.assertIn("Excerpt (method): We follow DeiT and here.", result["context"])

    def test_excerpt_is_best_body_section_never_references(self):
        paper = self.papers[0]
        db.session.add_all(
            [
                # Extractors store the abstract at order_index 0; it must not win the excerpt slot.
                PaperSection(paper_id=paper.id, section_type="abstract", text="We propose a thing.", order_index=0),
                PaperSection(paper_id=paper.id, section_type="method", text="We lift features to 3D.", order_index=1),
                PaperSection(paper_id=paper.id, section_type="references", text="[1] Voxel nets.", order_index=9),
            ]
        )
        db.session.commit()
        with _patch_embeddings(FakeEmbeddingService()):
            result = rag.retrieve_saved_context("voxel", top_k=6, paper_ids=[paper.id])

        self.assertEqual(result["sources"][0]["section"], "method")
        self.assertIn("Excerpt (method): We lift features to 3D.", result["context"])
        self.assertNotIn("Voxel nets", result["context"])


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
        self.assertEqual(result["scope"], "saved")
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["query"], "what is new in segmentation?")

    def test_llm_enabled_returns_synthesis(self):
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded answer citing RAG Paper 0."))]
        )
        fake_client = SimpleNamespace(complete=lambda **kwargs: fake_response)

        with (
            patch("app.services.rag.search_hybrid", return_value=self.ranked),
            patch("app.services.rag._build_client", return_value=fake_client),
        ):
            result = rag.answer_query("summarize my saved work", app=self.app)

        self.assertEqual(result["synthesis"], "Grounded answer citing RAG Paper 0.")
        self.assertTrue(result["llm_used"])
        self.assertEqual(len(result["sources"]), 2)

    def test_fabricated_citation_is_stripped(self):
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Real [2], fake [7]."))]
        )
        fake_client = SimpleNamespace(complete=lambda **kwargs: fake_response)
        with patch("app.services.rag._build_client", return_value=fake_client):
            result = rag.answer_query("anything", app=self.app)

        self.assertEqual(len(result["sources"]), 2)
        self.assertIn("[2]", result["synthesis"])
        self.assertNotIn("[7]", result["synthesis"])

    def test_llm_failure_degrades_to_none(self):
        def _boom(**kwargs):
            raise RuntimeError("network down")

        fake_client = SimpleNamespace(complete=_boom)
        with (
            patch("app.services.rag.search_hybrid", return_value=self.ranked),
            patch("app.services.rag._build_client", return_value=fake_client),
        ):
            result = rag.answer_query("anything", app=self.app)

        self.assertIsNone(result["synthesis"])
        self.assertFalse(result["llm_used"])

    def test_synthesize_uses_throttled_complete_not_private_helper(self):
        # Regression: _synthesize must go through the throttled public complete() so chat
        # respects max_concurrent, not reach into the private _create_completion.
        from unittest.mock import Mock

        client = Mock()
        client.complete.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )
        out = rag._synthesize(client, "q", "ctx")
        self.assertEqual(out, "answer")
        client.complete.assert_called_once()
        client._create_completion.assert_not_called()

    def test_no_saved_papers_answers_from_whole_corpus(self):
        PaperFeedback.query.delete()
        db.session.commit()

        with patch("app.services.rag.search_hybrid", return_value=self.ranked):
            result = rag.answer_query("anything at all", app=self.app)
        self.assertEqual(result["scope"], "corpus")
        self.assertEqual([s["paper_id"] for s in result["sources"]], [p.id for p in self.papers[:2]])
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
        # Add an UNSAVED paper and have hybrid surface it first: with saves present
        # chat is scoped to them, so the unsaved paper must not appear.
        saved = Paper.query.one()
        unsaved = _make_paper(1)
        db.session.add(unsaved)
        db.session.commit()
        ranked = [
            {"paper_id": unsaved.id, "rrf_score": 0.9, "bm25_rank": 1, "semantic_rank": 1},
            {"paper_id": saved.id, "rrf_score": 0.5, "bm25_rank": 2, "semantic_rank": 2},
        ]
        with patch("app.services.rag.search_hybrid", return_value=ranked):
            response = self.client.post(
                "/api/corpus/chat",
                json={"query": "what did I save?"},
                headers={"X-CSRF-Token": self.csrf_token},
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["scope"], "saved")
        self.assertFalse(data["llm_used"])
        self.assertIsNone(data["synthesis"])
        source_ids = [s["paper_id"] for s in data["sources"]]
        self.assertEqual(source_ids, [saved.id])
        self.assertNotIn(unsaved.id, source_ids)

    def test_chat_scoped_to_collection(self):
        saved = Paper.query.one()
        member = _make_paper(1)
        collection = Collection(name="Review")
        db.session.add_all([member, collection])
        db.session.flush()
        db.session.add(PaperCollection(paper_id=member.id, collection_id=collection.id))
        db.session.commit()

        response = self.client.post(
            "/api/corpus/chat",
            json={"query": "anything", "collection_id": collection.id},
            headers={"X-CSRF-Token": self.csrf_token},
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["scope"], "collection")
        self.assertEqual([s["paper_id"] for s in data["sources"]], [member.id])
        self.assertNotIn(saved.id, [s["paper_id"] for s in data["sources"]])

    def test_chat_collection_skips_hidden_members(self):
        member, skipped = _make_paper(1), _make_paper(2, is_hidden=True)
        collection = Collection(name="Review")
        db.session.add_all([member, skipped, collection])
        db.session.flush()
        db.session.add_all([PaperCollection(paper_id=p.id, collection_id=collection.id) for p in (member, skipped)])
        db.session.commit()

        response = self.client.post(
            "/api/corpus/chat",
            json={"query": "anything", "collection_id": collection.id},
            headers={"X-CSRF-Token": self.csrf_token},
        )

        self.assertEqual([s["paper_id"] for s in response.get_json()["sources"]], [member.id])

    def test_chat_rejects_unknown_or_malformed_collection(self):
        headers = {"X-CSRF-Token": self.csrf_token}
        missing = self.client.post("/api/corpus/chat", json={"query": "q", "collection_id": 999}, headers=headers)
        self.assertEqual(missing.status_code, 404)
        malformed = self.client.post("/api/corpus/chat", json={"query": "q", "collection_id": "1"}, headers=headers)
        self.assertEqual(malformed.status_code, 400)

    def test_discover_sends_collection_from_url(self):
        collection = Collection(name="Occupancy <Review>")
        db.session.add(collection)
        db.session.commit()

        text = self.client.get(f"/discover?collection={collection.id}").get_data(as_text=True)

        self.assertIn(f"const collectionId = {collection.id};", text)
        self.assertIn("collection_id: collectionId", text)
        self.assertIn("Occupancy &lt;Review&gt;", text)
        self.assertIn('id="chat-scope"', text)
        self.assertIn("const collectionId = null;", self.client.get("/discover").get_data(as_text=True))
