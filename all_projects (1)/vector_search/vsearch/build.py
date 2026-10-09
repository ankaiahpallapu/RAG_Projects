"""Build a sharded index from a float16/float32 .npy matrix (rows = vectors, row number = vector id).

    python -m vsearch.build --data data/vectors.npy --out index --shards 8

Build time is dominated by IVF k-means training. For very large corpora run one process per machine with
`--only 0,1` / `--only 2,3` ... against shared storage; shards are fully independent.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import IndexConfig
from .shard import Shard


def build_index(
    data_path: str,
    out_dir: str,
    cfg: IndexConfig,
    only_shards: list[int] | None = None,
    seed: int = 0,
    log=print,
) -> IndexConfig:
    data = np.load(data_path, mmap_mode="r")
    n, dim = data.shape
    cfg = dataclasses.replace(cfg, dim=dim)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    S = cfg.num_shards
    shard_ids = list(range(S)) if only_shards is None else only_shards

    # Phase 1: train each shard's coarse quantizer + PQ codebooks on a random sample of *its* vectors.
    shards: dict[int, Shard] = {}
    for s in shard_ids:
        t0 = time.time()
        n_s = len(range(s, n, S))  # ids with id % S == s
        if cfg.backend == "faiss" and n_s:
            n_train = min(cfg.train_size, n_s)
            rng = np.random.default_rng(seed + s)
            picks = np.sort(rng.choice(n_s, size=n_train, replace=False))
            train = np.asarray(data[s + S * picks], dtype=np.float32)
        else:
            train = np.empty((0, dim), np.float32)
        shards[s] = Shard.create(s, cfg, out, train, n_expected=n_s)
        log(f"[train] shard {s}: {n_s:,} vectors, nlist={shards[s].nlist}, {time.time() - t0:.1f}s")

    # Phase 2: ONE sequential pass over the data, routing row i to shard (i % S).
    t0 = time.time()
    block = cfg.add_batch
    for start in range(0, n, block):
        stop = min(n, start + block)
        chunk = np.asarray(data[start:stop], dtype=np.float32)
        gids = np.arange(start, stop, dtype=np.int64)
        for s, shard in shards.items():
            sel = slice((s - start) % S, None, S)
            shard.add(chunk[sel], gids[sel])
        if (start // block) % 10 == 0 or stop == n:
            rate = stop / max(time.time() - t0, 1e-9)
            log(f"[add] {stop:,}/{n:,} rows ({rate:,.0f} rows/s)")
    for shard in shards.values():
        shard.save()

    manifest = {
        "config": dataclasses.asdict(cfg),
        "n_vectors": n,
        "shard_sizes": [len(range(s, n, S)) for s in range(S)],
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done: {n:,} vectors in {S} shards -> {out}")
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help=".npy matrix, shape (N, dim)")
    p.add_argument("--out", required=True, help="index directory")
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--backend", choices=["faiss", "flat"], default="faiss")
    p.add_argument("--pq-m", type=int, help="PQ sub-quantizers (default dim/8)")
    p.add_argument("--pq-bits", type=int, default=8)
    p.add_argument("--nlist", type=int, help="IVF cells per shard (default ~4*sqrt(n))")
    p.add_argument("--train-size", type=int, default=500_000, help="training vectors per shard")
    p.add_argument("--no-raw", action="store_true", help="skip float16 originals (no exact re-ranking, less disk)")
    p.add_argument("--nprobe", type=int, default=24, help="default query nprobe stored in the manifest")
    p.add_argument("--rerank-factor", type=int, default=4)
    p.add_argument("--only", help="comma-separated shard ids to build on this machine")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    cfg = IndexConfig(
        num_shards=a.shards,
        backend=a.backend,
        pq_m=a.pq_m,
        pq_bits=a.pq_bits,
        nlist=a.nlist,
        train_size=a.train_size,
        store_raw=not a.no_raw,
        default_nprobe=a.nprobe,
        default_rerank_factor=a.rerank_factor,
    )
    only = [int(x) for x in a.only.split(",")] if a.only else None
    build_index(a.data, a.out, cfg, only, a.seed)


if __name__ == "__main__":
    main()
