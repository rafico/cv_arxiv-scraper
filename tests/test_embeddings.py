"""Tests for SPECTER2 embeddings + the exact inner-product vector index."""

from __future__ import annotations

import json
import logging
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flask import has_app_context

from app.services.embeddings import (
    EmbeddingService,
    add_papers_to_index,
    add_sections_to_index,
    get_embedding_service,
    reset_embedding_service,
)


def _fake_encode(texts, **kwargs):
    vecs = np.random.default_rng(7).random((len(texts), 768)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _unit_vec():
    v = np.full((768,), 0.1, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_load_reconciles_index_map_drift(tmp_path):
    # A torn write (crash between save()'s two os.replace calls) can leave the index and
    # id_map disagreeing on length. On load they must be reconciled to the consistent
    # prefix, not silently operated on as a drifted pair.
    index_dir = tmp_path / "faiss_index"
    svc = EmbeddingService(str(index_dir))
    svc.add_papers([1, 2, 3], ["a", "b", "c"], vectors=[_unit_vec(), _unit_vec(), _unit_vec()])
    svc.save()

    # index has 3 vectors; simulate an id_map that lost its last entry.
    (index_dir / "id_map.json").write_text("[1, 2]")

    reloaded = EmbeddingService(str(index_dir))
    assert reloaded.index_count() == 2
    assert reloaded._id_map == [1, 2]
    assert reloaded.has_paper(1) and not reloaded.has_paper(3)


def test_partial_state_does_not_clobber_surviving_index(tmp_path):
    # If only papers.npy survives (id_map lost), a fresh service must run degraded and
    # NOT overwrite the good index file on the next save — otherwise a recoverable
    # partial state becomes total vector loss.
    index_dir = tmp_path / "faiss_index"
    svc = EmbeddingService(str(index_dir))
    svc.add_papers([1, 2, 3], ["a", "b", "c"], vectors=[_unit_vec(), _unit_vec(), _unit_vec()])
    svc.save()

    index_path = index_dir / "papers.npy"
    original = index_path.read_bytes()
    (index_dir / "id_map.json").unlink()  # lose the map only

    degraded = EmbeddingService(str(index_dir))
    assert degraded.index_count() == 0  # read-empty degraded mode
    degraded.add_papers([9], ["z"], vectors=[_unit_vec()])
    degraded.save()  # must be a no-op for the main index

    assert index_path.read_bytes() == original  # the 3-vector index was preserved

    # Once the pair on disk is whole again, saving is back on: the row added meanwhile goes on top.
    (index_dir / "id_map.json").write_text("[1, 2, 3]")
    degraded.save()
    assert json.loads((index_dir / "id_map.json").read_text()) == [1, 2, 3, 9]


def _axis_vec(axis: int) -> np.ndarray:
    """A vector that names its paper: 1.0 in the paper id's component."""
    v = np.zeros(768, dtype=np.float32)
    v[axis] = 1.0
    return v


def _other_process_adds(index_dir, *paper_ids: int) -> None:
    # What the scrape (another process) does: load the index, add, save.
    add_papers_to_index(
        str(index_dir), list(paper_ids), [""] * len(paper_ids), vectors=[_axis_vec(i) for i in paper_ids]
    )


def test_reload_and_save_follow_other_writers(tmp_path, caplog):
    from app.services import embeddings

    def rereads(service) -> bool:
        with patch.object(embeddings, "_read_matrix") as read_matrix:
            service.reload_if_changed()
        return read_matrix.called

    index_dir = tmp_path / "faiss_index"
    reader = EmbeddingService(index_dir)  # a long-running web or MCP server, started on an empty index
    _other_process_adds(index_dir, 1, 2)

    reader.reload_if_changed()
    assert reader.index_count() == 2 and reader.has_paper(2)
    assert not rereads(reader)  # loaded once: until the files change again a check is one stat

    # Its own save is nothing to reload: no file is read again.
    reader.add_papers([3], [""], vectors=[_axis_vec(3)])
    reader.save()
    assert not rereads(reader)

    # A row it has not saved yet stays through a reload, on top of someone else's save.
    reader.add_papers([4], [""], vectors=[_axis_vec(4)])
    _other_process_adds(index_dir, 5)
    reader.reload_if_changed()
    assert reader.has_paper(4) and reader.has_paper(5)

    # A save from a stale index builds on what was saved meanwhile instead of writing over it.
    stale = EmbeddingService(index_dir)
    _other_process_adds(index_dir, 6, 7)
    stale.add_papers([8, 6], ["", ""], vectors=[_axis_vec(8), _axis_vec(6)])  # 6 was embedded by both: kept once
    stale.save()
    assert json.loads((index_dir / "id_map.json").read_text()) == [1, 2, 3, 5, 6, 7, 8]
    reader.save()  # still holds 4, and has seen neither 6, 7 nor 8
    on_disk = EmbeddingService(index_dir)
    found, vectors = on_disk.get_paper_vectors(list(range(1, 9)))
    assert vectors.argmax(axis=1).tolist() == found == list(range(1, 9))  # all there, each row its own paper's

    # A torn pair (the map already names a row the matrix does not have yet) is left
    # alone: read once, reported once, and loaded as soon as either file makes it agree.
    matrix = np.load(index_dir / "papers.npy")
    id_map = json.loads((index_dir / "id_map.json").read_text())
    (index_dir / "id_map.json").write_text(json.dumps([*id_map, 9]))
    with caplog.at_level(logging.WARNING, logger="app.services.embeddings"):
        on_disk.reload_if_changed()
        assert on_disk.index_count() == 8 and not on_disk.has_paper(9)
        assert not rereads(on_disk)
    assert caplog.text.count("not a matching pair") == 1
    np.save(index_dir / "papers.npy", np.vstack([matrix, _axis_vec(9)]))
    on_disk.reload_if_changed()
    assert on_disk.has_paper(9)
    # A file that cannot be read at all is left alone the same way.
    (index_dir / "id_map.json").write_text("{not json")
    on_disk.reload_if_changed()
    assert on_disk.index_count() == 9 and not rereads(on_disk)


def test_no_app_fallback_follows_the_instance_path_override(tmp_path, monkeypatch):
    # With no app in reach the singleton must land where create_app() would put it (the
    # sandbox tests/conftest.py sets), not in <cwd>/instance: from a checkout that is real data.
    monkeypatch.chdir(tmp_path)  # so that a regression makes its ./instance here
    monkeypatch.delenv("FAISS_INDEX_DIR", raising=False)
    monkeypatch.setenv("CV_ARXIV_INSTANCE_PATH", str(tmp_path / "sandbox"))
    assert not has_app_context()

    assert get_embedding_service().index_dir == (tmp_path / "sandbox" / "faiss_index").resolve()


def test_ensure_section_index_loads_once(tmp_path):
    # The section index uses double-checked locking: once loaded, a second call must
    # take the fast path and not re-read the index off disk (or clobber in-memory state).
    from app.services import embeddings

    svc = EmbeddingService(str(tmp_path / "faiss_index"))
    svc._ensure_section_index()
    first = svc._section_index
    with patch.object(embeddings, "_read_matrix") as read_matrix:
        svc._ensure_section_index()
    read_matrix.assert_not_called()
    assert svc._section_index is first


def test_legacy_faiss_index_migrates_on_load(tmp_path):
    # A pre-0.5 index dir holds faiss-format papers.index/sections.index files.
    # First load with faiss importable must convert to .npy with identical vectors
    # and leave the legacy file untouched for rollback.
    faiss = pytest.importorskip("faiss")

    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    vectors = _fake_encode(["a", "b", "c"])
    legacy = faiss.IndexFlatIP(768)
    legacy.add(vectors)
    faiss.write_index(legacy, str(index_dir / "papers.index"))
    (index_dir / "id_map.json").write_text("[11, 22, 33]")
    legacy_bytes = (index_dir / "papers.index").read_bytes()

    svc = EmbeddingService(str(index_dir))

    assert svc.index_count() == 3
    assert svc.has_paper(22)
    found, migrated = svc.get_paper_vectors([11, 22, 33])
    assert found == [11, 22, 33]
    np.testing.assert_allclose(migrated, vectors, rtol=1e-6)
    assert (index_dir / "papers.npy").exists()
    assert (index_dir / "papers.index").read_bytes() == legacy_bytes

    # A reload now takes the .npy path (no faiss involved) and sees the same data.
    reloaded = EmbeddingService(str(index_dir))
    assert reloaded.index_count() == 3


def test_legacy_index_without_faiss_runs_read_empty_and_protects_files(tmp_path):
    # Legacy .index present but faiss not importable: run empty with save disabled
    # so the legacy file (and its id_map) survive for migration or rebuild.
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    (index_dir / "papers.index").write_bytes(b"legacy-faiss-bytes")
    (index_dir / "id_map.json").write_text("[1]")

    import builtins

    real_import = builtins.__import__

    def no_faiss(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("No module named 'faiss'")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=no_faiss):
        svc = EmbeddingService(str(index_dir))

    assert svc.index_count() == 0
    svc.add_papers([9], ["z"], vectors=[_unit_vec()])
    svc.save()  # must not write papers.npy or touch the legacy pair
    assert not (index_dir / "papers.npy").exists()
    assert (index_dir / "papers.index").read_bytes() == b"legacy-faiss-bytes"
    assert (index_dir / "id_map.json").read_text() == "[1]"


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

    def test_paper_indexed_while_it_was_being_encoded_gets_no_second_row(self, index_dir):
        # Encoding runs outside the lock, so a reload or another thread's add can index
        # the same paper meanwhile.
        service = EmbeddingService(index_dir)

        def encode_while_another_add_lands(texts):
            service.add_papers([1], [""], vectors=[_axis_vec(1)])
            return np.stack([_axis_vec(2)] * len(texts))

        with patch.object(EmbeddingService, "encode", side_effect=encode_while_another_add_lands):
            assert service.add_papers([1], ["a"]) == 0
        assert service._id_map == [1]

    def test_save_and_reload(self, index_dir):
        service = _make_service(index_dir)
        service.add_papers([10, 20], ["hello world", "foo bar"])
        service.save()

        # Verify files exist
        assert (index_dir / "papers.npy").exists()
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

    def test_load_model_is_thread_safe_and_constructs_once(self, index_dir, monkeypatch):
        # Concurrent cold-start search requests must not each load the ~400MB model.
        service = EmbeddingService(index_dir)
        monkeypatch.setattr("app.services.embeddings.EMBEDDING_MODEL_CANDIDATES", ("the-model",))

        construct_count = 0
        count_lock = threading.Lock()

        def slow_loader(model_name):
            nonlocal construct_count
            with count_lock:
                construct_count += 1
            time.sleep(0.1)  # widen the race window
            return MagicMock()

        with patch("sentence_transformers.SentenceTransformer", side_effect=slow_loader):
            threads = [threading.Thread(target=service._load_model) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert construct_count == 1
        assert service._model is not None
