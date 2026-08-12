"""SPECTER2 embeddings + an exact inner-product vector index for similarity and search.

The index is a plain float32 matrix persisted as ``papers.npy`` (sections:
``sections.npy``) beside the unchanged ``id_map.json`` sidecars. It replaced a
faiss ``IndexFlatIP``: search was always exact, so a NumPy matmul does the same
work without the heavyweight dependency (or its dual-libgomp SIGSEGV
mitigations). Legacy ``*.index`` files are migrated on first load when faiss is
still importable; otherwise ``cv-arxiv-backfill --rebuild-index`` rebuilds.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None

import numpy as np
from flask import current_app, has_app_context

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

# Cross-process file locks (one per resolved index dir), keyed so every caller in
# this process shares the same reentrant object.
_FILE_LOCKS: dict[str, _ReentrantFileLock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


class _ReentrantFileLock:
    """Reentrant exclusive lock spanning threads (RLock) and processes (fcntl.flock).

    The in-process ``threading.Lock`` on ``EmbeddingService`` only serializes threads
    within one process, so a CLI backfill/rebuild run against a live server's index dir
    can race the scrape writer and clobber one side's vectors (whichever ``os.replace``
    lands last wins). This lock serializes the whole on-disk load-modify-save sequence
    across processes too. Reentrant so nested acquisitions in one thread
    (add_papers_to_index -> save -> save_sections) don't self-deadlock.
    """

    def __init__(self, lock_path: Path):
        self._lock_path = lock_path
        self._rlock = threading.RLock()
        self._depth = 0
        self._fd: int | None = None

    @contextmanager
    def acquire(self):
        self._rlock.acquire()
        entered = False
        try:
            if self._depth == 0 and fcntl is not None:
                fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                except Exception:
                    os.close(fd)
                    raise
                self._fd = fd
            self._depth += 1
            entered = True
            yield
        finally:
            if entered:
                self._depth -= 1
                if self._depth == 0 and self._fd is not None:
                    try:
                        fcntl.flock(self._fd, fcntl.LOCK_UN)
                    finally:
                        os.close(self._fd)
                        self._fd = None
            self._rlock.release()


def _index_file_lock(index_dir: str | Path) -> _ReentrantFileLock:
    key = str(Path(index_dir).resolve())
    with _FILE_LOCKS_GUARD:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            # Sibling of the index dir (not inside it) so the lock file survives the dir
            # being (re)created/copied/swapped — and works before the dir exists. The
            # parent (instance dir) is guaranteed present.
            lock_path = Path(key).parent / (Path(key).name + ".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = _ReentrantFileLock(lock_path)
            _FILE_LOCKS[key] = lock
        return lock


class _FlatIndex:
    """Exact inner-product search over a float32 matrix.

    The subset of ``faiss.IndexFlatIP`` this app ever used (``add`` / ``search`` /
    ``reconstruct`` / ``reconstruct_n`` / ``ntotal``), so call sites are unchanged.
    Exact search over a single-user corpus (well under a million 768-dim rows) is
    a millisecond matmul — no approximate index structure needed.
    """

    def __init__(self, matrix: np.ndarray | None = None):
        self._matrix = matrix if matrix is not None else np.empty((0, DIMENSION), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._matrix.shape[0])

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def add(self, vectors: np.ndarray) -> None:
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        # ponytail: O(n) copy per batch add; switch to chunked growth if it ever hurts.
        self._matrix = np.vstack([self._matrix, vecs]) if self.ntotal else vecs.copy()

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k rows by inner product per query row: (scores, indices), both (q, k)."""
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if self.ntotal == 0 or k <= 0:
            empty = np.empty((query.shape[0], 0))
            return empty.astype(np.float32), empty.astype(np.int64)
        k = min(k, self.ntotal)
        scores = query @ self._matrix.T
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
        indices = np.take_along_axis(top, order, axis=1)
        return np.take_along_axis(scores, indices, axis=1), indices.astype(np.int64)

    def reconstruct(self, row: int) -> np.ndarray:
        return self._matrix[row].copy()

    def reconstruct_n(self, start: int, count: int) -> np.ndarray:
        return self._matrix[start : start + count].copy()


def _read_matrix(npy_path: Path, legacy_path: Path) -> np.ndarray | None:
    """Load the vector matrix from ``.npy``, migrating a legacy faiss file if present.

    Returns None when only a legacy index exists and faiss is not importable —
    the caller must then run read-empty with save disabled so the legacy file
    survives for a rebuild.
    """
    if npy_path.exists():
        matrix = np.load(npy_path)
        return np.asarray(matrix, dtype=np.float32).reshape(-1, DIMENSION)

    try:
        import faiss

        # The legacy dual-libgomp mitigation, scoped to this one-time migration:
        # keep faiss' OpenMP pool single-threaded so it can't race torch's.
        try:
            faiss.omp_set_num_threads(1)
        except Exception:  # pragma: no cover - older faiss builds
            pass
    except ImportError:
        LOGGER.error(
            "Legacy faiss index at %s but faiss is not installed; starting empty. "
            "Either `pip install faiss-cpu` once to auto-migrate, or rebuild with "
            "`cv-arxiv-backfill --rebuild-index`.",
            legacy_path,
        )
        return None

    legacy = faiss.read_index(str(legacy_path))
    matrix = legacy.reconstruct_n(0, legacy.ntotal) if legacy.ntotal else np.empty((0, DIMENSION))
    matrix = np.asarray(matrix, dtype=np.float32).reshape(-1, DIMENSION)
    _save_matrix_atomic(matrix, npy_path)  # legacy file left untouched for rollback
    LOGGER.info("Migrated legacy faiss index %s -> %s (%d vectors)", legacy_path.name, npy_path.name, len(matrix))
    return matrix


def _save_matrix_atomic(matrix: np.ndarray, npy_path: Path) -> None:
    tmp_path = str(npy_path) + ".tmp"
    with open(tmp_path, "wb") as f:
        np.save(f, np.ascontiguousarray(matrix, dtype=np.float32))
    os.replace(tmp_path, str(npy_path))


class EmbeddingService:
    """Manages SPECTER2 embeddings and the on-disk vector index."""

    def __init__(self, index_dir: str | Path):
        self._index_dir = Path(index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self._index_dir / "papers.npy"
        self._legacy_index_path = self._index_dir / "papers.index"
        self._id_map_path = self._index_dir / "id_map.json"

        self._model = None
        self._index: _FlatIndex | None = None
        # Maps index row position -> paper PK
        self._id_map: list[int] = []
        # Reverse: paper PK -> index row position
        self._pk_to_row: dict[int, int] = {}
        self._lock = threading.Lock()
        # Separate from _lock so a (slow, one-time) model load doesn't serialize
        # with index search/add.
        self._model_lock = threading.Lock()

        self._load_index()

    def _load_index(self) -> None:
        # Persisting is safe unless we detect a corrupt/partial on-disk pair below, in
        # which case save() must NOT overwrite the surviving file with an empty/drifted
        # index — that would turn a recoverable partial state into total data loss.
        self._persistable = True

        index_exists = self._index_path.exists() or self._legacy_index_path.exists()
        map_exists = self._id_map_path.exists()

        if not index_exists and not map_exists:
            self._index = _FlatIndex()
            self._id_map = []
            self._pk_to_row = {}
            return

        if index_exists != map_exists:
            # Exactly one sidecar survived (a crash between save()'s two renames, or
            # external file loss). The missing half can't be reconstructed here, so run
            # in a degraded read-empty mode and DISABLE save — the survivor is left
            # intact for a proper rebuild (cv-arxiv-backfill --rebuild-index).
            LOGGER.error(
                "Vector index in a partial state (index=%s, id_map.json=%s); starting "
                "empty and disabling save to avoid clobbering the survivor. Rebuild to recover.",
                index_exists,
                map_exists,
            )
            self._index = _FlatIndex()
            self._id_map = []
            self._pk_to_row = {}
            self._persistable = False
            return

        # Both present: load, then reconcile any length drift (a torn write leaves the
        # index and id_map disagreeing) down to their consistent prefix so the pair can
        # never silently mis-map rows or persist a drifted state on the next save().
        matrix = _read_matrix(self._index_path, self._legacy_index_path)
        if matrix is None:
            self._index = _FlatIndex()
            self._id_map = []
            self._pk_to_row = {}
            self._persistable = False
            return
        self._index = _FlatIndex(matrix)
        with open(self._id_map_path) as f:
            self._id_map = json.load(f)

        n = self._index.ntotal
        m = len(self._id_map)
        if n != m:
            keep = min(n, m)
            LOGGER.error(
                "Vector index/id_map drift (index=%d, map=%d); reconciling to %d consistent rows",
                n,
                m,
                keep,
            )
            if n > keep:
                self._index = self._prefix_index(self._index, keep)
            self._id_map = self._id_map[:keep]

        self._pk_to_row = {pk: row for row, pk in enumerate(self._id_map)}
        LOGGER.info("Loaded vector index with %d vectors", self._index.ntotal)

    @staticmethod
    def _prefix_index(index: _FlatIndex, keep: int) -> _FlatIndex:
        """Return a new flat index holding only the first ``keep`` vectors of ``index``."""
        return _FlatIndex(index.reconstruct_n(0, keep) if keep > 0 else None)

    def _load_model(self):
        # Double-checked locking: without the lock, two concurrent cold-start
        # search requests would each load the ~400MB SPECTER2 model.
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
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
        """Add papers to the vector index. Returns count added.

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
        # orphaning an index row and inflating the count.
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
        """Number of vectors in the index. Cheap — does not load the model."""
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

    def sample_paper_vectors(
        self,
        count: int,
        exclude_ids: set[int] | None = None,
        seed: int | None = None,
    ) -> tuple[list[int], np.ndarray]:
        """Random sample of indexed papers and their vectors (without replacement).

        Used by the learned ranker to draw weak-negative corpus papers.
        ``exclude_ids`` (e.g. labeled papers) are never sampled. Deterministic
        when ``seed`` is given.
        """
        if count <= 0:
            return [], np.empty((0, DIMENSION), dtype=np.float32)

        with self._lock:
            exclude = exclude_ids or set()
            pool = [(pid, row) for row, pid in enumerate(self._id_map) if pid not in exclude]
            if not pool:
                return [], np.empty((0, DIMENSION), dtype=np.float32)
            rng = np.random.default_rng(seed)
            if len(pool) > count:
                chosen = rng.choice(len(pool), size=count, replace=False)
                pool = [pool[int(idx)] for idx in chosen]
            found_ids = [pid for pid, _row in pool]
            vectors = [self._index.reconstruct(row) for _pid, row in pool]

        return found_ids, np.asarray(vectors, dtype=np.float32)

    def _ensure_section_index(self) -> None:
        """Load or create the section-level vector index (double-checked locking).

        The (possibly slow) disk read happens OUTSIDE ``self._lock`` so a one-time
        cold-start section load can't freeze concurrent paper searches/adds that share
        ``self._lock``; the lock is held only for the brief in-memory assignment. Assign
        _section_id_map before _section_index so the unlocked fast-path check (hasattr
        _section_index) never sees a half-initialized pair. Mirrors _load_index's
        partial-survivor + drift handling so a torn save_sections() can't silently
        mis-map rows or clobber a surviving index.
        """
        if hasattr(self, "_section_index"):
            return

        section_index_path = self._index_dir / "sections.npy"
        legacy_section_path = self._index_dir / "sections.index"
        section_map_path = self._index_dir / "section_id_map.json"
        index_exists = section_index_path.exists() or legacy_section_path.exists()
        map_exists = section_map_path.exists()
        sections_persistable = True

        if not index_exists and not map_exists:
            section_index = _FlatIndex()
            section_id_map: list[dict] = []
        elif index_exists != map_exists:
            # One sidecar survived a crash between save_sections()'s two renames. The
            # missing half can't be reconstructed; run read-empty and DISABLE save so
            # the survivor is left intact for a rebuild instead of being overwritten.
            LOGGER.error(
                "Section index partial (sections index=%s, section_id_map.json=%s); "
                "starting empty and disabling section save to avoid clobbering the survivor.",
                index_exists,
                map_exists,
            )
            section_index = _FlatIndex()
            section_id_map = []
            sections_persistable = False
        else:
            matrix = _read_matrix(section_index_path, legacy_section_path)
            if matrix is None:
                section_index = _FlatIndex()
                section_id_map = []
                sections_persistable = False
            else:
                section_index = _FlatIndex(matrix)
                with open(section_map_path) as f:
                    section_id_map = json.load(f)
                n = section_index.ntotal
                m = len(section_id_map)
                if n != m:
                    keep = min(n, m)
                    LOGGER.error(
                        "Section index/id_map drift (index=%d, map=%d); reconciling to %d consistent rows",
                        n,
                        m,
                        keep,
                    )
                    if n > keep:
                        section_index = self._prefix_index(section_index, keep)
                    section_id_map = section_id_map[:keep]

        with self._lock:
            if hasattr(self, "_section_index"):
                return
            self._sections_persistable = sections_persistable
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
        """Persist the section index to disk."""
        if not hasattr(self, "_section_index"):
            return

        if not getattr(self, "_sections_persistable", True):
            # Loaded from a partial section-index state (see _ensure_section_index);
            # writing our empty/degraded index would clobber the surviving file.
            LOGGER.warning("Skipping section index save: loaded from a partial/corrupt state")
            return

        with _index_file_lock(self._index_dir).acquire(), self._lock:
            section_map_path = self._index_dir / "section_id_map.json"
            tmp_map = str(section_map_path) + ".tmp"

            _save_matrix_atomic(self._section_index.matrix, self._index_dir / "sections.npy")
            with open(tmp_map, "w") as f:
                json.dump(self._section_id_map, f)
            os.replace(tmp_map, str(section_map_path))

    def save(self) -> None:
        """Persist the vector index (and section index if loaded) to disk atomically."""
        self.save_sections()

        with _index_file_lock(self._index_dir).acquire(), self._lock:
            if not self._persistable:
                # Loaded from a partial/corrupt on-disk state (see _load_index). Writing
                # our empty/degraded in-memory index would clobber the surviving file.
                LOGGER.warning("Skipping vector index save: loaded from a partial/corrupt state")
                return

            tmp_map = str(self._id_map_path) + ".tmp"

            _save_matrix_atomic(self._index.matrix, self._index_path)
            with open(tmp_map, "w") as f:
                json.dump(self._id_map, f)
            os.replace(tmp_map, str(self._id_map_path))

    def has_paper(self, paper_id: int) -> bool:
        # Guard the read: add_papers() mutates _pk_to_row under _lock, so an unlocked
        # read can observe an in-flux mapping.
        with self._lock:
            return paper_id in self._pk_to_row

    def index_count(self) -> int:
        """Alias of :meth:`index_size` kept for existing callers (search/related/ranking)."""
        return self.index_size()

    @property
    def index_dir(self) -> Path:
        return self._index_dir


def add_papers_to_index(index_dir: str, paper_ids: list[int], texts: list[str], vectors: list | None = None) -> int:
    """Load the on-disk index, add papers, and persist. Importable + dependency-free
    (no Flask/DB) so it can run in an isolated subprocess via run_isolated()."""
    # Hold the cross-process index lock across load (constructor) + modify + save so a
    # concurrent CLI/scrape writer can't interleave and drop vectors.
    with _index_file_lock(index_dir).acquire():
        service = EmbeddingService(index_dir)
        added = service.add_papers(paper_ids, texts, vectors=vectors)
        if added:
            service.save()
    return added


def add_sections_to_index(index_dir: str, entries: list[tuple[int, str, str]]) -> int:
    """Load the on-disk section index, add section embeddings, and persist. Importable +
    dependency-free (no Flask/DB) so it can run in an isolated subprocess via
    run_isolated() — mirrors add_papers_to_index for the torch-heavy section path."""
    with _index_file_lock(index_dir).acquire():
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


def peek_embedding_service() -> EmbeddingService | None:
    """Return the singleton if it already exists, WITHOUT creating it.

    Lets lightweight callers (learned-ranker artifact resolution) locate the
    index dir from scrape worker threads with no app context, while never
    triggering an index load as a side effect.
    """
    return _service_instance


def reset_embedding_service() -> None:
    """Reset the singleton (for testing)."""
    global _service_instance
    with _service_lock:
        _service_instance = None
