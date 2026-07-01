"""SPECTER2 embeddings + FAISS vector index for paper similarity and search."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

# faiss-cpu and torch each bundle their own copy of libgomp (GNU OpenMP). Loading
# two OpenMP runtimes into one process corrupts the heap (observed as SIGSEGV /
# "malloc(): unaligned tcache chunk detected" during the scrape embedding stage).
# Pin OpenMP to a single thread before faiss/torch (imported lazily below) load.
# Overridable via the env var; the thread count is also applied programmatically
# in _load_index()/_load_model() as defence in depth.
os.environ.setdefault("OMP_NUM_THREADS", "1")
_OMP_THREADS = max(1, int(os.environ.get("OMP_NUM_THREADS", "1") or "1"))

import numpy as np  # noqa: E402
from flask import current_app, has_app_context  # noqa: E402

LOGGER = logging.getLogger(__name__)

DIMENSION = 768
EMBEDDING_MODEL_CANDIDATES = tuple(
    dict.fromkeys(
        model
        for model in (
            os.environ.get("CV_ARXIV_EMBEDDING_MODEL"),
            "allenai/specter2_base",
            "allenai/specter",
        )
        if model
    )
)

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
        # Separate from _lock so a (slow, one-time) model load doesn't serialize
        # with FAISS index search/add.
        self._model_lock = threading.Lock()

        self._load_index()

    def _load_index(self) -> None:
        import faiss

        # Keep faiss' OpenMP pool single-threaded to avoid the dual-libgomp crash.
        try:
            faiss.omp_set_num_threads(_OMP_THREADS)
        except Exception:  # pragma: no cover - older faiss builds
            pass

        # Persisting is safe unless we detect a corrupt/partial on-disk pair below, in
        # which case save() must NOT overwrite the surviving file with an empty/drifted
        # index — that would turn a recoverable partial state into total data loss.
        self._persistable = True

        index_exists = self._index_path.exists()
        map_exists = self._id_map_path.exists()

        if not index_exists and not map_exists:
            self._index = faiss.IndexFlatIP(DIMENSION)
            self._id_map = []
            self._pk_to_row = {}
            return

        if index_exists != map_exists:
            # Exactly one sidecar survived (a crash between save()'s two renames, or
            # external file loss). The missing half can't be reconstructed here, so run
            # in a degraded read-empty mode and DISABLE save — the survivor is left
            # intact for a proper rebuild (cv-arxiv-backfill --rebuild-index).
            LOGGER.error(
                "FAISS index in a partial state (papers.index=%s, id_map.json=%s); starting "
                "empty and disabling save to avoid clobbering the survivor. Rebuild to recover.",
                index_exists,
                map_exists,
            )
            self._index = faiss.IndexFlatIP(DIMENSION)
            self._id_map = []
            self._pk_to_row = {}
            self._persistable = False
            return

        # Both present: load, then reconcile any length drift (a torn write leaves the
        # index and id_map disagreeing) down to their consistent prefix so the pair can
        # never silently mis-map rows or persist a drifted state on the next save().
        self._index = faiss.read_index(str(self._index_path))
        with open(self._id_map_path) as f:
            self._id_map = json.load(f)

        n = self._index.ntotal
        m = len(self._id_map)
        if n != m:
            keep = min(n, m)
            LOGGER.error(
                "FAISS index/id_map drift (index=%d, map=%d); reconciling to %d consistent rows",
                n,
                m,
                keep,
            )
            if n > keep:
                self._index = self._prefix_index(self._index, keep)
            self._id_map = self._id_map[:keep]

        self._pk_to_row = {pk: row for row, pk in enumerate(self._id_map)}
        LOGGER.info("Loaded FAISS index with %d vectors", self._index.ntotal)

    @staticmethod
    def _prefix_index(index, keep: int):
        """Return a new flat index holding only the first ``keep`` vectors of ``index``."""
        import faiss

        new_index = faiss.IndexFlatIP(DIMENSION)
        if keep > 0:
            new_index.add(index.reconstruct_n(0, keep))
        return new_index

    def _load_model(self):
        # Double-checked locking: without the lock, two concurrent cold-start
        # search requests would each load the ~400MB SPECTER2 model.
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            try:
                import torch

                # Match torch' OpenMP pool to faiss' so the two libgomp copies don't race.
                torch.set_num_threads(_OMP_THREADS)
            except Exception:  # pragma: no cover - torch optional/absent
                pass
            from sentence_transformers import SentenceTransformer

            last_exc: Exception | None = None
            for model_name in EMBEDDING_MODEL_CANDIDATES:
                try:
                    LOGGER.info("Loading embedding model %s (first call may download model weights)...", model_name)
                    self._model = SentenceTransformer(model_name)
                    LOGGER.info("Embedding model loaded: %s", model_name)
                    return
                except Exception as exc:
                    last_exc = exc
                    LOGGER.warning("Failed to load embedding model %s: %s", model_name, exc)

            raise RuntimeError("Unable to load any embedding model") from last_exc

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into L2-normalized embeddings."""
        self._load_model()
        embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)

    def add_papers(self, paper_ids: list[int], texts: list[str], vectors: list | None = None) -> int:
        """Add papers to the FAISS index. Returns count added.

        `vectors` may carry precomputed (L2-normalized) embeddings aligned with
        `paper_ids`; entries that are None are encoded from the matching text.
        """
        if not paper_ids:
            return 0

        aligned_vectors = list(vectors) if vectors is not None else [None] * len(paper_ids)

        # Filter out papers already indexed. Track ids accepted in THIS call too:
        # _pk_to_row is only updated after the add loop below, so a paper_ids list
        # containing the same id twice (e.g. a paper cross-listed across two RSS
        # feeds, not deduped between feeds) would otherwise add the vector twice —
        # orphaning a FAISS row and inflating the count.
        new_ids = []
        new_texts = []
        new_vectors = []
        seen: set[int] = set()
        for pid, text, vec in zip(paper_ids, texts, aligned_vectors):
            if pid not in self._pk_to_row and pid not in seen:
                new_ids.append(pid)
                new_texts.append(text)
                new_vectors.append(vec)
                seen.add(pid)

        if not new_ids:
            return 0

        to_encode = [idx for idx, vec in enumerate(new_vectors) if vec is None]
        if to_encode:
            encoded = self.encode([new_texts[idx] for idx in to_encode])
            for encoded_idx, idx in enumerate(to_encode):
                new_vectors[idx] = encoded[encoded_idx]
        embeddings = np.asarray(new_vectors, dtype=np.float32)

        with self._lock:
            self._index.add(embeddings)
            for pid in new_ids:
                self._pk_to_row[pid] = len(self._id_map)
                self._id_map.append(pid)

        return len(new_ids)

    def index_size(self) -> int:
        """Number of vectors in the FAISS index. Cheap — does not load the model."""
        with self._lock:
            return int(self._index.ntotal)

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

    def get_paper_vectors(self, paper_ids: list[int]) -> tuple[list[int], np.ndarray]:
        """Return indexed paper IDs and their reconstructed embedding vectors."""
        if not paper_ids or self._index.ntotal == 0:
            return [], np.empty((0, DIMENSION), dtype=np.float32)

        found_ids: list[int] = []
        vectors: list[np.ndarray] = []

        with self._lock:
            for paper_id in paper_ids:
                row = self._pk_to_row.get(paper_id)
                if row is None:
                    continue
                found_ids.append(paper_id)
                vectors.append(self._index.reconstruct(row))

        if not found_ids:
            return [], np.empty((0, DIMENSION), dtype=np.float32)

        return found_ids, np.asarray(vectors, dtype=np.float32)

    def _ensure_section_index(self) -> None:
        """Load or create the section-level FAISS index (double-checked locking).

        Mirrors _load_model: without the lock, two concurrent search_sections callers
        could each read the index off disk and one overwrite the other's in-memory
        state. Assign _section_id_map before _section_index so the unlocked fast-path
        check (hasattr _section_index) never sees a half-initialized pair.
        """
        if hasattr(self, "_section_index"):
            return

        import faiss

        with self._lock:
            if hasattr(self, "_section_index"):
                return
            section_index_path = self._index_dir / "sections.index"
            section_map_path = self._index_dir / "section_id_map.json"
            if section_index_path.exists() and section_map_path.exists():
                section_index = faiss.read_index(str(section_index_path))
                with open(section_map_path) as f:
                    section_id_map = json.load(f)
            else:
                section_index = faiss.IndexFlatIP(DIMENSION)
                section_id_map = []
            self._section_id_map = section_id_map
            self._section_index = section_index

    def add_sections(
        self,
        entries: list[tuple[int, str, str]],
    ) -> int:
        """Add section-level embeddings to the section index.

        Args:
            entries: list of (paper_id, section_type, text) tuples.

        Returns count of sections added.
        """
        if not entries:
            return 0

        self._ensure_section_index()

        # Skip papers already represented in the section index. Unlike add_papers,
        # this index has no removal path, so re-embedding an already-indexed paper
        # (e.g. it is re-scraped) would append duplicate section vectors — bloating
        # the index and returning the same paper multiple times from search_sections.
        # Dedup is per *paper*, not per row: one paper legitimately contributes many
        # section rows (intro/method/…) in a single call, so we must NOT drop a
        # paper's later sections here (that is the add_papers one-vector-per-id case).
        indexed_paper_ids = {m["paper_id"] for m in self._section_id_map}
        fresh = [(pid, stype, text) for pid, stype, text in entries if pid not in indexed_paper_ids]
        if not fresh:
            return 0

        texts = [text for _, _, text in fresh]
        meta = [{"paper_id": pid, "section_type": stype} for pid, stype, _ in fresh]

        embeddings = self.encode(texts)

        with self._lock:
            self._section_index.add(embeddings)
            self._section_id_map.extend(meta)

        return len(fresh)

    def search_sections(
        self,
        query_text: str,
        top_k: int = 20,
        section_type: str | None = None,
    ) -> list[dict]:
        """Search section-level embeddings.

        Returns list of dicts with paper_id, section_type, score.
        """
        self._ensure_section_index()
        if self._section_index.ntotal == 0:
            return []

        query_vec = self.encode([query_text])

        with self._lock:
            # Search more than needed if filtering by type.
            search_k = min(top_k * 3 if section_type else top_k, self._section_index.ntotal)
            scores, indices = self._section_index.search(query_vec, search_k)
            map_snapshot = list(self._section_id_map)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(map_snapshot):
                continue
            entry = map_snapshot[idx]
            if section_type and entry["section_type"] != section_type:
                continue
            results.append(
                {
                    "paper_id": entry["paper_id"],
                    "section_type": entry["section_type"],
                    "score": float(score),
                }
            )
            if len(results) >= top_k:
                break

        return results

    def save_sections(self) -> None:
        """Persist section FAISS index to disk."""
        import faiss

        if not hasattr(self, "_section_index"):
            return

        with self._lock:
            section_index_path = self._index_dir / "sections.index"
            section_map_path = self._index_dir / "section_id_map.json"

            tmp_index = str(section_index_path) + ".tmp"
            tmp_map = str(section_map_path) + ".tmp"

            faiss.write_index(self._section_index, tmp_index)
            with open(tmp_map, "w") as f:
                json.dump(self._section_id_map, f)

            os.replace(tmp_index, str(section_index_path))
            os.replace(tmp_map, str(section_map_path))

    def save(self) -> None:
        """Persist FAISS index (and section index if loaded) to disk atomically."""
        self.save_sections()
        import faiss

        with self._lock:
            if not self._persistable:
                # Loaded from a partial/corrupt on-disk state (see _load_index). Writing
                # our empty/degraded in-memory index would clobber the surviving file.
                LOGGER.warning("Skipping FAISS index save: loaded from a partial/corrupt state")
                return

            tmp_index = str(self._index_path) + ".tmp"
            tmp_map = str(self._id_map_path) + ".tmp"

            faiss.write_index(self._index, tmp_index)
            with open(tmp_map, "w") as f:
                json.dump(self._id_map, f)

            os.replace(tmp_index, str(self._index_path))
            os.replace(tmp_map, str(self._id_map_path))

    def has_paper(self, paper_id: int) -> bool:
        # Guard the read: add_papers() mutates _pk_to_row under _lock, so an unlocked
        # read can observe an in-flux mapping.
        with self._lock:
            return paper_id in self._pk_to_row

    def index_count(self) -> int:
        with self._lock:
            return self._index.ntotal

    @property
    def index_dir(self) -> Path:
        return self._index_dir


def add_papers_to_index(index_dir: str, paper_ids: list[int], texts: list[str], vectors: list | None = None) -> int:
    """Load the on-disk index, add papers, and persist. Importable + dependency-free
    (no Flask/DB) so it can run in an isolated subprocess via run_isolated()."""
    service = EmbeddingService(index_dir)
    added = service.add_papers(paper_ids, texts, vectors=vectors)
    if added:
        service.save()
    return added


def add_sections_to_index(index_dir: str, entries: list[tuple[int, str, str]]) -> int:
    """Load the on-disk section index, add section embeddings, and persist. Importable +
    dependency-free (no Flask/DB) so it can run in an isolated subprocess via
    run_isolated() — mirrors add_papers_to_index for the torch/faiss section path."""
    service = EmbeddingService(index_dir)
    added = service.add_sections(entries)
    if added:
        service.save_sections()
    return added


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
        elif has_app_context():
            # Several request/scrape-context callers pass no app. Prefer the active
            # app's configured index dir over the env/CWD fallback, which can diverge
            # under a non-default CWD or instance path.
            index_dir = current_app.config.get(
                "FAISS_INDEX_DIR",
                str(Path(current_app.instance_path) / "faiss_index"),
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
