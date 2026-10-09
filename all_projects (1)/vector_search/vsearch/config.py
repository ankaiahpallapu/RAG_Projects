from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class IndexConfig:
    """Build-time configuration; persisted in manifest.json so a loaded index is self-describing."""

    dim: int = 384
    num_shards: int = 8
    backend: str = "faiss"  # "faiss": IVF-HNSW,PQ  |  "flat": exact NumPy search (tests / small data)

    # --- IVF-PQ (faiss backend) ---
    pq_m: int | None = None  # sub-quantizers; None -> dim // 8  (384 dims -> 48 bytes/vector)
    pq_bits: int = 8
    nlist: int | None = None  # coarse cells per shard; None -> ~4*sqrt(vectors in shard), power of two
    hnsw_m: int = 32  # HNSW graph over the coarse centroids (fast cell selection)
    quantizer_ef_search: int = 64
    train_size: int = 500_000  # max training vectors per shard

    # --- exact re-ranking ---
    store_raw: bool = True  # keep float16 originals on disk to re-score PQ candidates exactly

    # --- query defaults ---
    default_nprobe: int = 24
    default_rerank_factor: int = 4  # fetch k*factor PQ candidates, re-score exactly, keep k

    add_batch: int = 200_000

    def resolve_pq_m(self) -> int:
        m = self.pq_m or max(1, self.dim // 8)
        if self.dim % m:
            raise ValueError(f"dim={self.dim} must be divisible by pq_m={m}; set pq_m explicitly")
        return m

    def resolve_nlist(self, n_vectors: int, n_train: int) -> int:
        if self.nlist:
            nlist = self.nlist
        else:
            nlist = 1 << max(4, round(math.log2(max(2.0, 4 * math.sqrt(max(n_vectors, 1))))))
        # FAISS wants >= 39 training points per centroid for stable k-means
        return max(1, min(nlist, n_train // 39))
