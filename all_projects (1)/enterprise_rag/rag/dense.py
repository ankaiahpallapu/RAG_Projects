from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # FAISS is optional: without it we fall back to exact NumPy search.
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


class DenseIndex:
    """Inner-product index over L2-normalised embeddings (== cosine similarity).

    backend="faiss": HNSW graph, sub-linear search, scales to millions of chunks.
    backend="numpy": exact brute force, zero dependencies, fine up to ~100K chunks.
    Vector ids are insertion order (0..N-1) and line up with the BM25 index and chunk store.
    """

    def __init__(
        self,
        dim: int,
        backend: str | None = None,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ):
        self.dim = dim
        self.backend = backend or ("faiss" if faiss is not None else "numpy")
        self.ef_search = ef_search
        self._parts: list[np.ndarray] = []
        self._mat: np.ndarray | None = None
        self._index = None
        if self.backend == "faiss":
            if faiss is None:
                raise ImportError("backend='faiss' requested but faiss is not installed")
            self._index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efConstruction = ef_construction
            self._index.hnsw.efSearch = ef_search
        elif self.backend != "numpy":
            raise ValueError(f"unknown dense backend {self.backend!r}")

    @property
    def ntotal(self) -> int:
        if self.backend == "faiss":
            return int(self._index.ntotal)
        return sum(len(p) for p in self._parts) if self._mat is None else len(self._mat)

    def add(self, vecs: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        if self.backend == "faiss":
            self._index.add(vecs)
        else:
            self._parts.append(vecs)
            self._mat = None

    def _matrix(self) -> np.ndarray:
        if self._mat is None:
            self._mat = np.vstack(self._parts) if self._parts else np.empty((0, self.dim), np.float32)
            self._parts = [self._mat]
        return self._mat

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, ids), each shape (nq, k'), best first, where k' = min(k, ntotal)."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        k = min(k, self.ntotal)
        if k == 0:
            n = len(queries)
            return np.empty((n, 0), np.float32), np.empty((n, 0), np.int64)
        if self.backend == "faiss":
            return self._index.search(queries, k)
        sims = queries @ self._matrix().T
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        part = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-part, axis=1)
        return np.take_along_axis(part, order, axis=1), np.take_along_axis(top, order, axis=1).astype(np.int64)

    # ---- persistence -------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        if self.backend == "faiss":
            faiss.write_index(self._index, str(d / "dense.faiss"))
        else:
            np.save(d / "dense.npy", self._matrix())

    @classmethod
    def load(cls, directory: str | Path, backend: str, dim: int, ef_search: int = 128) -> "DenseIndex":
        d = Path(directory)
        idx = cls.__new__(cls)
        idx.dim, idx.backend, idx.ef_search = dim, backend, ef_search
        idx._parts, idx._mat, idx._index = [], None, None
        if backend == "faiss":
            if faiss is None:
                raise ImportError("this index was built with faiss; `pip install faiss-cpu` to load it")
            idx._index = faiss.read_index(str(d / "dense.faiss"))
            idx._index.hnsw.efSearch = ef_search
        else:
            idx._mat = np.load(d / "dense.npy")
            idx._parts = [idx._mat]
        return idx
