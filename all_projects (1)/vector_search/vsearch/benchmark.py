"""Measure recall and latency against exact ground truth.

    python -m vsearch.benchmark --index index --data data/vectors.npy --queries 200

Reports, for a sweep of (rerank_factor, nprobe):  recall@k vs exact search, and single-query latency
p50/p95/p99.  Then a concurrent-load test (QPS and tail latency) at the chosen operating point.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .cluster import ShardedIndex
from .datagen import make_queries
from .exact import exact_topk


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    hits = [len(set(f[:k].tolist()) & set(t[:k].tolist())) for f, t in zip(found, truth)]
    return float(np.mean(hits)) / k


def ground_truth(queries: np.ndarray, data: np.ndarray, k: int, cache: str | None) -> np.ndarray:
    if cache and Path(cache).exists():
        cached = np.load(cache)
        if cached.shape == (len(queries), k):
            return cached
    t0 = time.time()
    _, ids = exact_topk(queries, data, k)
    print(f"ground truth over {len(data):,} vectors computed in {time.time() - t0:.1f}s")
    if cache:
        np.save(cache, ids)
    return ids


def sweep(index: ShardedIndex, queries: np.ndarray, truth: np.ndarray, k: int, nprobes: list[int], rerank_factors: list[int]):
    rows = []
    for rf in rerank_factors:
        for nprobe in nprobes:
            lat = np.empty(len(queries))
            found = np.full((len(queries), k), -1, np.int64)
            for i in range(len(queries)):
                t0 = time.perf_counter()
                _, ids = index.search(queries[i : i + 1], k, nprobe=nprobe, rerank_factor=rf)
                lat[i] = (time.perf_counter() - t0) * 1000
                found[i] = ids[0]
            rows.append((rf, nprobe, recall_at_k(found, truth, k), *np.percentile(lat, [50, 95, 99])))
    return rows


def load_test(index: ShardedIndex, queries: np.ndarray, k: int, nprobe: int, rf: int, concurrency: int, total: int):
    lat: list[float] = []

    def one(i: int) -> float:
        t0 = time.perf_counter()
        index.search(queries[i % len(queries) : i % len(queries) + 1], k, nprobe=nprobe, rerank_factor=rf)
        return (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        lat = list(pool.map(one, range(total)))
    wall = time.perf_counter() - t0
    return total / wall, *np.percentile(lat, [50, 99])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", required=True)
    p.add_argument("--data", required=True, help="the .npy used to build the index (for exact ground truth)")
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--nprobe", default="8,16,32,64")
    p.add_argument("--rerank", default="0,4", help="rerank factors to compare (0 = PQ scores only)")
    p.add_argument("--concurrency", default="1,8,32")
    p.add_argument("--truth-cache", help="path to cache ground-truth ids (.npy) between runs")
    p.add_argument("--omp-threads", type=int, default=1)
    p.add_argument("--target-ms", type=float, default=100.0, help="p99 latency budget used to pick the operating point")
    a = p.parse_args()

    index = ShardedIndex.load(a.index, omp_threads=a.omp_threads)
    data = np.load(a.data, mmap_mode="r")
    queries = make_queries(data, a.queries)
    truth = ground_truth(queries, data, a.k, a.truth_cache)
    print(f"index: {index.stats()}\n")

    for i in range(min(20, len(queries))):  # warm caches / page in mmap
        index.search(queries[i : i + 1], a.k)

    nprobes = [int(x) for x in a.nprobe.split(",")]
    rfs = [int(x) for x in a.rerank.split(",")]
    rows = sweep(index, queries, truth, a.k, nprobes, rfs)
    print(f"{'rerank':>7}{'nprobe':>8}{'recall@' + str(a.k):>11}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}")
    for rf, nprobe, rec, p50, p95, p99 in rows:
        print(f"{rf:>7}{nprobe:>8}{rec:>11.3f}{p50:>9.2f}{p95:>9.2f}{p99:>9.2f}")

    ok = [r for r in rows if r[5] <= a.target_ms]
    if not ok:
        print(f"\nno configuration met p99 <= {a.target_ms} ms; lower nprobe or add shards/cores")
        return
    rf, nprobe, rec, *_ = max(ok, key=lambda r: r[2])
    print(f"\noperating point (best recall with p99 <= {a.target_ms:g} ms): rerank={rf} nprobe={nprobe} recall={rec:.3f}")
    print(f"\n{'threads':>8}{'QPS':>10}{'p50 ms':>9}{'p99 ms':>9}")
    for c in [int(x) for x in a.concurrency.split(",")]:
        qps, p50, p99 = load_test(index, queries, a.k, nprobe, rf, c, total=max(300, c * 40))
        print(f"{c:>8}{qps:>10.0f}{p50:>9.2f}{p99:>9.2f}")


if __name__ == "__main__":
    main()
