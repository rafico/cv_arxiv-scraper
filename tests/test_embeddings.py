"""Tests for SPECTER2 embeddings + FAISS vector index."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.embeddings import EmbeddingService, reset_embedding_service


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
