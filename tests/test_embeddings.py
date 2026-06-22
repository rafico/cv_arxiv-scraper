"""Tests for SPECTER2 embeddings + FAISS vector index."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.embeddings import EmbeddingService, add_sections_to_index, reset_embedding_service


def _fake_encode(texts, **kwargs):
    vecs = np.random.default_rng(7).random((len(texts), 768)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_add_sections_to_index_persists_round_trip(tmp_path):
    """The isolated section-embedding helper must persist to disk so the parent can
    reload the singleton after the subprocess writes it (mirrors add_papers_to_index)."""
    index_dir = tmp_path / "faiss_index"
    entries = [(1, "method", "a method section"), (2, "results", "the results")]
    with patch.object(EmbeddingService, "encode", side_effect=_fake_encode):
        added = add_sections_to_index(str(index_dir), entries)
        assert added == 2
        # A fresh service must read the section index the helper persisted.
        reloaded = EmbeddingService(str(index_dir))
        hits = reloaded.search_sections("method", top_k=5)
    assert hits  # non-empty: the persisted sections are searchable


def test_importing_embeddings_pins_openmp_to_avoid_dual_libgomp_crash():
    # faiss-cpu and torch each bundle libgomp; loading both with multithreaded
    # OpenMP corrupts the heap. Importing the module must pin OMP to a thread count.
    from app.services import embeddings

    assert os.environ.get("OMP_NUM_THREADS")
    assert embeddings._OMP_THREADS >= 1


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_embedding_service()
    yield
    reset_embedding_service()


@pytest.fixture
def index_dir(tmp_path):
    return tmp_path / "faiss_index"


def _make_service(index_dir, dim=768):
    """Create an EmbeddingService with a mocked model."""
    service = EmbeddingService(index_dir)
    mock_model = MagicMock()

    def fake_encode(texts, **kwargs):
        vecs = np.random.default_rng(42).random((len(texts), dim)).astype(np.float32)
        # L2 normalise to match real behaviour
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock_model.encode = fake_encode
    service._model = mock_model
    return service


class TestEmbeddingService:
    def test_add_and_search(self, index_dir):
        service = _make_service(index_dir)

        paper_ids = [1, 2, 3]
        texts = ["neural network image classification", "object detection with transformers", "graph neural networks"]
        added = service.add_papers(paper_ids, texts)

        assert added == 3
        assert service.index_count() == 3
        assert service.has_paper(1)
        assert service.has_paper(2)
        assert not service.has_paper(999)

    def test_add_papers_with_precomputed_vectors_skips_encode(self, index_dir):
        service = EmbeddingService(index_dir)
        # No model attached: any encode attempt would try to load the real model.
        vectors = np.eye(3, 768, dtype=np.float32)

        added = service.add_papers([1, 2, 3], ["a", "b", "c"], vectors=list(vectors))

        assert added == 3
        found_ids, reconstructed = service.get_paper_vectors([1, 2, 3])
        assert found_ids == [1, 2, 3]
        np.testing.assert_allclose(reconstructed, vectors, atol=1e-6)

    def test_add_papers_encodes_only_missing_vectors(self, index_dir):
        service = _make_service(index_dir)
        precomputed = np.zeros(768, dtype=np.float32)
        precomputed[0] = 1.0

        added = service.add_papers([1, 2], ["a", "b"], vectors=[precomputed, None])

        assert added == 2
        found_ids, reconstructed = service.get_paper_vectors([1])
        np.testing.assert_allclose(reconstructed[0], precomputed, atol=1e-6)

    def test_search_returns_results(self, index_dir):
        service = _make_service(index_dir)
        service.add_papers([10, 20, 30], ["alpha", "beta", "gamma"])

        results = service.search("alpha query", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(pid, int) and isinstance(score, float) for pid, score in results)

    def test_search_by_id(self, index_dir):
        service = _make_service(index_dir)
        service.add_papers([1, 2, 3, 4], ["a", "b", "c", "d"])

        results = service.search_by_id(1, top_k=2)
        assert len(results) <= 2
        assert all(pid != 1 for pid, _ in results)

    def test_search_by_id_unknown(self, index_dir):
        service = _make_service(index_dir)
        service.add_papers([1], ["test"])
        assert service.search_by_id(999) == []

    def test_get_paper_vectors_returns_indexed_ids_in_requested_order(self, index_dir):
        service = _make_service(index_dir)
        service.add_papers([1, 2, 3], ["alpha", "beta", "gamma"])

        paper_ids, vectors = service.get_paper_vectors([3, 999, 1])

        assert paper_ids == [3, 1]
        assert vectors.shape == (2, 768)

    def test_no_duplicate_adds(self, index_dir):
        service = _make_service(index_dir)
        service.add_papers([1, 2], ["a", "b"])
        added = service.add_papers([2, 3], ["b", "c"])
        assert added == 1
        assert service.index_count() == 3

    def test_save_and_reload(self, index_dir):
        service = _make_service(index_dir)
        service.add_papers([10, 20], ["hello world", "foo bar"])
        service.save()

        # Verify files exist
        assert (index_dir / "papers.index").exists()
        assert (index_dir / "id_map.json").exists()

        # Load a new service from same dir
        service2 = EmbeddingService(index_dir)
        assert service2.index_count() == 2
        assert service2.has_paper(10)
        assert service2.has_paper(20)

    def test_empty_index_search(self, index_dir):
        service = _make_service(index_dir)
        assert service.search("query") == []
        assert service.search_by_id(1) == []

    def test_add_empty_list(self, index_dir):
        service = _make_service(index_dir)
        assert service.add_papers([], []) == 0

    def test_load_model_falls_back_to_next_candidate(self, index_dir, monkeypatch):
        service = EmbeddingService(index_dir)
        monkeypatch.setattr(
            "app.services.embeddings.EMBEDDING_MODEL_CANDIDATES",
            ("broken-model", "working-model"),
        )

        loaded_models = []

        def fake_loader(model_name):
            loaded_models.append(model_name)
            if model_name == "broken-model":
                raise RuntimeError("bad model")
            return MagicMock()

        with patch("sentence_transformers.SentenceTransformer", side_effect=fake_loader):
            service._load_model()

        assert loaded_models == ["broken-model", "working-model"]
        assert service._model is not None
