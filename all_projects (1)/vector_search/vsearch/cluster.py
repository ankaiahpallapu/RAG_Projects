from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .config import IndexConfig
from .exact import exact_topk, normalize, topk_pad
from .shard import Shard

MANIFEST = "manifest.json"


class DeltaBuffer:
    """In-memory, exactly-searched buffer for vectors added after the shards were built.

    New vectors are searchable immediately (real-time freshness) without touching the immutable,
    memory-mapped shard files. Intended to hold up to ~100K vectors between offline rebuilds.
    Copy-on-write, so concurrent searches always see a consistent snapshot.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._vecs = np.empty((0, dim), np.float32)
        self._ids = np.empty(0, np.int64)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        vecs = normalize(vecs)
        with self._lock:
            keep = ~np.isin(self._ids, ids)  # re-upserting an id replaces it
            self._vecs = np.concatenate([self._vecs[keep], vecs])
            self._ids = np.concatenate([self._ids[keep], ids])

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        vecs, ids = self._vecs, self._ids  # atomic snapshot
        return exact_topk(q, vecs, k, ids=ids)

    def save(self, path: str | Path) -> None:
        np.savez(path, ids=self._ids, vecs=self._vecs)

    def load(self, path: str | Path) -> None:
        data = np.load(path)
        self._ids, self._vecs = data["ids"], data["vecs"]


class ShardedIndex:
    """Fan-out/merge over hash-partitioned shards (id % num_shards).

    A query is sent to every shard in parallel (FAISS releases the GIL), each returns its exact top-k
    after re-ranking, and the global top-k is a cheap merge of num_shards*k candidates.
    """

    def __init__(self, root: str | Path, cfg: IndexConfig, shards: dict[int, Shard], n_vectors: int, omp_threads: int = 1):
        self.root, self.cfg, self.shards, self.n_vectors = Path(root), cfg, shards, n_vectors
        self.delta = DeltaBuffer(cfg.dim)
        self._omp_threads = omp_threads
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, min(len(shards), (os.cpu_count() or 4) * 2)),
            thread_name_prefix="shard",
            initializer=self._init_worker if cfg.backend == "faiss" else None,
        )

    def _init_worker(self) -> None:
        # OpenMP thread count is per calling thread. One OMP thread per shard task means the
        # parallelism comes from the fan-out and cores aren't oversubscribed.
        import faiss  # type: ignore

        faiss.omp_set_num_threads(self._omp_threads)

    @classmethod
    def load(
        cls,
        root: str | Path,
        shard_ids: list[int] | None = None,
        mmap: bool = True,
        omp_threads: int = 1,
    ) -> "ShardedIndex":
        root = Path(root)
        manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        cfg = IndexConfig(**manifest["config"])
        wanted = shard_ids if shard_ids is not None else list(range(cfg.num_shards))
        shards = {s: Shard.load(s, cfg, root, mmap=mmap) for s in wanted}
        index = cls(root, cfg, shards, manifest["n_vectors"], omp_threads)
        delta_path = root / "delta.npz"
        if delta_path.exists():
            index.delta.load(delta_path)
        return index

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        nprobe: int | None = None,
        rerank_factor: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (cosine scores, ids), each (nq, k), best first. Missing results are (-inf, -1)."""
        q = normalize(np.atleast_2d(queries))
        if q.shape[1] != self.cfg.dim:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.cfg.dim}")
        nprobe = nprobe or self.cfg.default_nprobe
        rf = self.cfg.default_rerank_factor if rerank_factor is None else rerank_factor

        if len(self.shards) == 1:
            results = [next(iter(self.shards.values())).search(q, k, nprobe, rf)]
        else:
            futures = [self._pool.submit(sh.search, q, k, nprobe, rf) for sh in self.shards.values()]
            results = [f.result() for f in futures]
        scores = np.concatenate([r[0] for r in results], axis=1)
        ids = np.concatenate([r[1] for r in results], axis=1)
        if len(self.delta):
            ds, di = self.delta.search(q, k)
            scores, ids = np.concatenate([scores, ds], axis=1), np.concatenate([ids, di], axis=1)
        return topk_pad(scores, ids, k)

    def upsert(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        """Make new vectors searchable immediately. Ids must be new (>= the built range)."""
        ids = np.asarray(ids, dtype=np.int64)
        if np.any(ids < self.n_vectors):
            raise ValueError(
                f"ids below {self.n_vectors} belong to the built shards and can't be updated in place; "
                "use new ids, or rebuild"
            )
        self.delta.add(ids, vecs)

    def persist_delta(self) -> None:
        self.delta.save(self.root / "delta.npz")

    def stats(self) -> dict:
        return {
            "backend": self.cfg.backend,
            "dim": self.cfg.dim,
            "shards_loaded": sorted(self.shards),
            "num_shards": self.cfg.num_shards,
            "vectors_in_loaded_shards": sum(s.n for s in self.shards.values()),
            "vectors_built_total": self.n_vectors,
            "delta_vectors": len(self.delta),
            "default_nprobe": self.cfg.default_nprobe,
            "default_rerank_factor": self.cfg.default_rerank_factor,
        }

    def close(self) -> None:
        self._pool.shutdown(wait=False)
