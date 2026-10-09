"""Synthetic embedding generator: unit vectors drawn from a Gaussian mixture, written as a float16 .npy memmap.

Mixture structure gives IVF cells something to latch onto, like real embeddings. It is still synthetic:
always tune nprobe / rerank_factor on a sample of your *real* embeddings before trusting the numbers.

    python -m vsearch.datagen --out data/vectors.npy --n 1000000 --dim 384
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .exact import normalize


def generate(path: str, n: int, dim: int = 384, clusters: int = 2048, noise: float = 0.6, seed: int = 0, chunk: int = 500_000) -> None:
    rng = np.random.default_rng(seed)
    centers = normalize(rng.standard_normal((clusters, dim)))
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, dim))
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        assign = rng.integers(0, clusters, m)
        x = centers[assign] + (noise / np.sqrt(dim)) * rng.standard_normal((m, dim)).astype(np.float32)
        out[start : start + m] = normalize(x).astype(np.float16)
    out.flush()


def make_queries(data: np.ndarray, n_queries: int, noise: float = 0.3, seed: int = 1) -> np.ndarray:
    """Queries = random corpus vectors plus noise, i.e. near (but not identical to) indexed points."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(data), size=n_queries, replace=False))
    base = np.asarray(data[rows], dtype=np.float32)
    dim = base.shape[1]
    return normalize(base + (noise / np.sqrt(dim)) * rng.standard_normal(base.shape).astype(np.float32))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--clusters", type=int, default=2048)
    p.add_argument("--noise", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    t0 = time.time()
    generate(a.out, a.n, a.dim, a.clusters, a.noise, a.seed)
    print(f"wrote {a.n:,} x {a.dim} float16 vectors to {a.out} in {time.time() - t0:.1f}s ({a.n * a.dim * 2 / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
