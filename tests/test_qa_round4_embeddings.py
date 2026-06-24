"""QA round 4 regression tests for app/services/embeddings.py.

G13: add_papers/add_sections must not add the same paper id twice within a single
     call (the persisted reverse map is only updated after the loop).
G18: get_embedding_service(app=None) must prefer the active Flask app's configured
     FAISS_INDEX_DIR over the env/CWD fallback.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from app.services.embeddings import (
    EmbeddingService,
    get_embedding_service,
    reset_embedding_service,
)
from tests.helpers import FlaskDBTestCase


def _make_service(index_dir, dim=768):
    service = EmbeddingService(index_dir)
    mock_model = MagicMock()

    def fake_encode(texts, **kwargs):
        vecs = np.random.default_rng(42).random((len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock_model.encode = fake_encode
    service._model = mock_model
    return service


def test_g13_add_papers_dedupes_same_id_within_single_call(tmp_path):
    service = _make_service(tmp_path / "faiss_index")

    added = service.add_papers([2, 2], ["dup", "dup"])

    assert added == 1
    assert service.index_count() == 1
    # Reverse map and id_map must agree with the FAISS row count.
    assert service.has_paper(2)
    assert len(service._id_map) == 1


def test_g13_add_sections_keeps_all_sections_of_one_paper_in_single_call(tmp_path):
    # add_sections dedups per *paper across calls*, not per row within a call: one
    # paper legitimately contributes many distinct section rows (intro/method/…) in a
    # single call and they must all be indexed. (Regression guard against a paper-level
    # intra-call dedup that would silently drop a paper's later sections — that is the
    # add_papers one-vector-per-id case, not this one.)
    service = _make_service(tmp_path / "faiss_index")

    added = service.add_sections([(2, "method", "a"), (2, "results", "b")])

    assert added == 2
    service._ensure_section_index()
    assert service._section_index.ntotal == 2
    assert len(service._section_id_map) == 2
    # Re-adding the same paper in a LATER call is deduped (already indexed).
    assert service.add_sections([(2, "method", "a"), (2, "results", "b")]) == 0


class GetEmbeddingServiceAppContextTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        reset_embedding_service()

    def tearDown(self):
        reset_embedding_service()
        super().tearDown()

    def test_g18_no_app_arg_resolves_config_index_dir_in_app_context(self):
        custom_dir = self._tmpdir.name + "/custom_faiss"
        self.app.config["FAISS_INDEX_DIR"] = custom_dir

        # No app argument: must resolve via the active app context, not env/CWD.
        service = get_embedding_service()

        self.assertEqual(str(service.index_dir), custom_dir)
