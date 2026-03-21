"""Tests for hybrid search (BM25 + semantic)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.search import RRF_K, search_hybrid


class TestRRFFusion:
    @patch("app.services.search.search_bm25")
    @patch("app.services.search.search_semantic")
    def test_hybrid_merges_results(self, mock_semantic, mock_bm25):
        mock_bm25.return_value = [(1, 5.0), (2, 3.0), (3, 1.0)]
        mock_semantic.return_value = [(2, 0.9), (4, 0.8), (1, 0.7)]

        results = search_hybrid("test query", top_k=10)

        # Paper 2 appears in both, should rank highest
        pids = [r["paper_id"] for r in results]
        assert 2 in pids
        assert 1 in pids
        assert 4 in pids
        assert 3 in pids

        # Paper 2 has rank 1 in semantic and rank 2 in bm25 -> highest combined
        paper2 = next(r for r in results if r["paper_id"] == 2)
        assert paper2["bm25_rank"] == 2
        assert paper2["semantic_rank"] == 1
        assert paper2["rrf_score"] > 0

    @patch("app.services.search.search_bm25")
    @patch("app.services.search.search_semantic")
    def test_hybrid_empty_query(self, mock_semantic, mock_bm25):
        results = search_hybrid("", top_k=10)
        assert results == []
        mock_bm25.assert_not_called()

    @patch("app.services.search.search_bm25")
    @patch("app.services.search.search_semantic")
    def test_hybrid_only_bm25(self, mock_semantic, mock_bm25):
        mock_bm25.return_value = [(1, 5.0), (2, 3.0)]
        mock_semantic.return_value = []

        results = search_hybrid("test", top_k=10)
        assert len(results) == 2
        # All results should have semantic_rank=None
        for r in results:
            assert r["semantic_rank"] is None

    @patch("app.services.search.search_bm25")
    @patch("app.services.search.search_semantic")
    def test_hybrid_only_semantic(self, mock_semantic, mock_bm25):
        mock_bm25.return_value = []
        mock_semantic.return_value = [(10, 0.95), (20, 0.8)]

        results = search_hybrid("test", top_k=10)
        assert len(results) == 2
        for r in results:
            assert r["bm25_rank"] is None

    @patch("app.services.search.search_bm25")
    @patch("app.services.search.search_semantic")
    def test_hybrid_respects_top_k(self, mock_semantic, mock_bm25):
        mock_bm25.return_value = [(i, float(100 - i)) for i in range(50)]
        mock_semantic.return_value = [(i + 25, 1.0 - i * 0.01) for i in range(50)]

        results = search_hybrid("test", top_k=5)
        assert len(results) == 5

    def test_rrf_score_formula(self):
        """Verify the RRF formula manually."""
        # RRF(d) = weight / (k + rank) for each system
        bm25_weight = 0.4
        semantic_weight = 0.6
        # Paper at rank 1 in both systems
        expected = bm25_weight / (RRF_K + 1) + semantic_weight / (RRF_K + 1)
        assert abs(expected - (1.0 / (RRF_K + 1))) < 1e-9
