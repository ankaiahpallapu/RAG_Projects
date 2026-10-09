from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import IndexConfig
from .exact import exact_topk, normalize


def _faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError('backend="faiss" needs `pip install faiss-cpu`') from exc
    return faiss


def _tune_quantizer(index, ef_search: int) -> None:
    """Set efSearch on the HNSW graph that selects which IVF cells to probe."""
    faiss = _faiss()
    ivf = faiss.extract_index_ivf(index)
    quantizer = faiss.downcast_index(ivf.quantizer)
    if hasattr(quantizer, "hnsw"):
        quantizer.hnsw.efSearch = ef_search


class Shard:
    """One partition of the corpus.

    faiss backend : IVF-HNSW,PQ index (~48 B/vector) + optional float16 originals for exact re-ranking.
    flat backend  : float16 originals only, searched exactly (for tests and small datasets).

    Vector ids are global int64 and must be added in strictly ascending order, which lets us map
    id -> row in the raw file with a binary search instead of a hash table.
    """

    def __init__(self, shard_id: int, dim: int, backend: str, root: str | Path, store_raw: bool = True):
        self.shard_id, self.dim, self.backend = shard_id, dim, backend
        self.root = Path(root)
        self.store_raw = store_raw or backend == "flat"
        self.index = None
        self.ids = np.empty(0, np.int64)
        self.raw: np.ndarray | None = None
        self.n = 0
        self.nlist: int | None = None
        self._id_parts: list[np.ndarray] = []
        self._last_id = -1
        self._raw_fh = None

    def _path(self, suffix: str) -> Path:
        return self.root / f"shard_{self.shard_id:03d}.{suffix}"

    # ------------------------------------------------------------------ build
    @classmethod
    def create(cls, shard_id: int, cfg: IndexConfig, root: str | Path, train_vecs: np.ndarray, n_expected: int) -> "Shard":
        shard = cls(shard_id, cfg.dim, cfg.backend, root, cfg.store_raw)
        if cfg.backend == "faiss":
            faiss = _faiss()
            train = normalize(train_vecs)
            shard.nlist = cfg.resolve_nlist(n_expected, len(train))
            factory = f"IVF{shard.nlist}_HNSW{cfg.hnsw_m},PQ{cfg.resolve_pq_m()}x{cfg.pq_bits}"
            # L2 on unit vectors ranks identically to cosine, and is the best-trodden IVF-PQ path.
            shard.index = faiss.index_factory(cfg.dim, factory, faiss.METRIC_L2)
            shard.index.train(train)
            _tune_quantizer(shard.index, cfg.quantizer_ef_search)
        if shard.store_raw:
            shard.root.mkdir(parents=True, exist_ok=True)
            shard._raw_fh = open(shard._path("raw"), "wb")
        return shard

    def add(self, vecs: np.ndarray, ids: np.ndarray) -> None:
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        if len(ids) == 0:
            return
        if ids[0] <= self._last_id or np.any(np.diff(ids) <= 0):
            raise ValueError("ids must be strictly ascending within and across add() calls")
        vecs = normalize(vecs)
        if self.index is not None:
            self.index.add_with_ids(vecs, ids)
        if self._raw_fh is not None:
            self._raw_fh.write(vecs.astype(np.float16).tobytes())
        self._id_parts.append(ids)
        self._last_id = int(ids[-1])
        self.n += len(ids)

    def save(self) -> None:
        if self._raw_fh is not None:
            self._raw_fh.close()
            self._raw_fh = None
        ids = np.concatenate(self._id_parts) if self._id_parts else np.empty(0, np.int64)
        np.save(self._path("ids.npy"), ids)
        if self.index is not None:
            _faiss().write_index(self.index, str(self._path("index")))

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, shard_id: int, cfg: IndexConfig, root: str | Path, mmap: bool = True) -> "Shard":
        shard = cls(shard_id, cfg.dim, cfg.backend, root, cfg.store_raw)
        shard.ids = np.load(shard._path("ids.npy"), mmap_mode="r" if mmap else None)
        shard.n = len(shard.ids)
        raw_path = shard._path("raw")
        if shard.n and raw_path.exists():
            shard.raw = np.memmap(raw_path, dtype=np.float16, mode="r", shape=(shard.n, cfg.dim))
        if cfg.backend == "faiss":
            faiss = _faiss()
            path = str(shard._path("index"))
            flags = getattr(faiss, "IO_FLAG_MMAP", 0) if mmap else 0
            try:  # memory-map the inverted lists: RAM use stays low and pages in on demand
                shard.index = faiss.read_index(path, flags)
            except Exception:  # noqa: BLE001 - fall back to a normal in-RAM load
                shard.index = faiss.read_index(path)
            _tune_quantizer(shard.index, cfg.quantizer_ef_search)
        return shard

    # ------------------------------------------------------------------ search
    def search(
        self, q: np.ndarray, k: int, nprobe: int = 24, rerank_factor: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """q: (nq, dim) unit vectors. Returns (cosine scores, global ids), shape (nq, k), best first."""
        nq = len(q)
        if self.n == 0:
            return np.full((nq, k), -np.inf, np.float32), np.full((nq, k), -1, np.int64)
        if self.backend == "flat":
            return exact_topk(q, self.raw, k, ids=self.ids)

        faiss = _faiss()
        rerank = bool(rerank_factor) and self.raw is not None
        kk = min(k * rerank_factor if rerank else k, self.n)
        params = faiss.SearchParametersIVF()
        params.nprobe = int(nprobe)  # per-call params: safe with concurrent requests
        dist, cand = self.index.search(q, kk, params=params)
        if rerank:
            return self._rerank(q, cand, k)
        scores = np.where(cand >= 0, 1.0 - dist / 2.0, -np.inf).astype(np.float32)  # unit vectors: cos = 1 - d^2/2
        return self._pad(scores[:, :k], cand[:, :k], k)

    def _rerank(self, q: np.ndarray, cand: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Re-score PQ candidates against the float16 originals: PQ finds the neighbourhood, exact math orders it."""
        nq = len(q)
        out_s = np.full((nq, k), -np.inf, np.float32)
        out_i = np.full((nq, k), -1, np.int64)
        for qi in range(nq):
            c = np.sort(cand[qi][cand[qi] >= 0])  # ascending ids -> ascending rows -> friendlier mmap reads
            if len(c) == 0:
                continue
            rows = np.searchsorted(self.ids, c)
            sims = self.raw[rows].astype(np.float32) @ q[qi]
            top = np.argsort(-sims)[:k]
            out_s[qi, : len(top)] = sims[top]
            out_i[qi, : len(top)] = c[top]
        return out_s, out_i

    @staticmethod
    def _pad(scores: np.ndarray, ids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        short = k - scores.shape[1]
        if short > 0:
            scores = np.concatenate([scores, np.full((len(scores), short), -np.inf, np.float32)], axis=1)
            ids = np.concatenate([ids, np.full((len(ids), short), -1, np.int64)], axis=1)
        return scores, ids
