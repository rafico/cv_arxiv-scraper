"""SPECTER2 embeddings + FAISS vector index for paper similarity and search."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

DIMENSION = 768
SPECTER2_MODEL = "allenai/specter2"

_service_instance: EmbeddingService | None = None
_service_lock = threading.Lock()


class EmbeddingService:
    """Manages SPECTER2 embeddings and a FAISS sidecar index."""

    def __init__(self, index_dir: str | Path):
        self._index_dir = Path(index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self._index_dir / "papers.index"
        self._id_map_path = self._index_dir / "id_map.json"

        self._model = None
        self._index = None
        # Maps FAISS row position -> paper PK
        self._id_map: list[int] = []
        # Reverse: paper PK -> FAISS row position
        self._pk_to_row: dict[int, int] = {}
        self._lock = threading.Lock()

        self._load_index()

    def _load_index(self) -> None:
        import faiss

        if self._index_path.exists() and self._id_map_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._id_map_path, "r") as f:
                self._id_map = json.load(f)
            self._pk_to_row = {pk: row for row, pk in enumerate(self._id_map)}
            LOGGER.info("Loaded FAISS index with %d vectors", self._index.ntotal)
        else:
            self._index = faiss.IndexFlatIP(DIMENSION)
            self._id_map = []
            self._pk_to_row = {}

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        LOGGER.info("Loading SPECTER2 model (first call may download ~420MB)...")
        self._model = SentenceTransformer(SPECTER2_MODEL)
        LOGGER.info("SPECTER2 model loaded")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into L2-normalized embeddings."""
        self._load_model()
        embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)

    def add_papers(self, paper_ids: list[int], texts: list[str]) -> int:
        """Generate embeddings and add to the FAISS index. Returns count added."""
        if not paper_ids:
            return 0

        # Filter out papers already indexed
        new_ids = []
        new_texts = []
        for pid, text in zip(paper_ids, texts):
            if pid not in self._pk_to_row:
                new_ids.append(pid)
                new_texts.append(text)

        if not new_ids:
            return 0

        embeddings = self.encode(new_texts)

        with self._lock:
            self._index.add(embeddings)
            for pid in new_ids:
                self._pk_to_row[pid] = len(self._id_map)
                self._id_map.append(pid)

        return len(new_ids)

    def search(self, query_text: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Search by text query. Returns [(paper_id, score)]."""
        if self._index.ntotal == 0:
            return []

        query_vec = self.encode([query_text])

        with self._lock:
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(query_vec, k)
            id_map_snapshot = list(self._id_map)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(id_map_snapshot):
                continue
            results.append((id_map_snapshot[idx], float(score)))
        return results

    def search_by_id(self, paper_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        """Find papers similar to an existing indexed paper."""
        with self._lock:
            row = self._pk_to_row.get(paper_id)
            if row is None or self._index.ntotal == 0:
                return []

            vec = self._index.reconstruct(row).reshape(1, -1)
            k = min(top_k + 1, self._index.ntotal)
            scores, indices = self._index.search(vec, k)
            id_map_snapshot = list(self._id_map)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(id_map_snapshot):
                continue
            pid = id_map_snapshot[idx]
            if pid == paper_id:
                continue
            results.append((pid, float(score)))
        return results[:top_k]

    def save(self) -> None:
        """Persist FAISS index to disk atomically."""
        import faiss

        with self._lock:
            tmp_index = str(self._index_path) + ".tmp"
            tmp_map = str(self._id_map_path) + ".tmp"

            faiss.write_index(self._index, tmp_index)
            with open(tmp_map, "w") as f:
                json.dump(self._id_map, f)

            os.replace(tmp_index, str(self._index_path))
            os.replace(tmp_map, str(self._id_map_path))

    def has_paper(self, paper_id: int) -> bool:
        return paper_id in self._pk_to_row

    def index_count(self) -> int:
        return self._index.ntotal


def get_embedding_service(app=None) -> EmbeddingService:
    """Return the singleton EmbeddingService, creating it if needed."""
    global _service_instance

    if _service_instance is not None:
        return _service_instance

    with _service_lock:
        if _service_instance is not None:
            return _service_instance

        if app is not None:
            index_dir = app.config.get(
                "FAISS_INDEX_DIR",
                str(Path(app.instance_path) / "faiss_index"),
            )
        else:
            index_dir = os.environ.get(
                "FAISS_INDEX_DIR",
                str(Path.cwd() / "instance" / "faiss_index"),
            )

        _service_instance = EmbeddingService(index_dir)
        return _service_instance


def reset_embedding_service() -> None:
    """Reset the singleton (for testing)."""
    global _service_instance
    with _service_lock:
        _service_instance = None
