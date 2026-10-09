"""Smoke tests.   python tests/smoke_test.py

The `flat` backend (exact NumPy search) needs nothing but NumPy, so it always runs and validates sharding,
id routing, merging, the real-time buffer, padding and metadata. If `faiss` is installed the IVF-PQ backend
is exercised too (build, mmap load, rerank, recall).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsearch.build import build_index  # noqa: E402
from vsearch.cluster import ShardedIndex  # noqa: E402
from vsearch.config import IndexConfig  # noqa: E402
from vsearch.datagen import generate, make_queries  # noqa: E402
from vsearch.exact import exact_topk, topk_pad  # noqa: E402
from vsearch.metadata import MetadataStore  # noqa: E402
from vsearch.shard import Shard  # noqa: E402

try:
    import faiss  # noqa: F401

    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False

N, DIM, K = 20_000, 64, 10


def _recall(found: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([len(set(f.tolist()) & set(t.tolist())) / len(t) for f, t in zip(found, truth)]))


def test_exact_and_pad():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1000, 8)).astype(np.float32)
    q = rng.standard_normal((3, 8)).astype(np.float32)
    s, i = exact_topk(q, data, 5, block=128)  # multiple blocks exercise the running merge
    full = q @ data.T
    assert (i == np.argsort(-full, axis=1)[:, :5]).all()
    s2, i2 = exact_topk(q, data[:3], 5)  # fewer rows than k -> padded
    assert (i2[:, 3:] == -1).all() and np.isneginf(s2[:, 3:]).all()
    s3, i3 = topk_pad(np.zeros((1, 2), np.float32), np.zeros((1, 2), np.int64), 4)
    assert i3.shape == (1, 4)
    print("ok  exact_topk / padding")


def test_shard_requires_ascending_ids():
    with tempfile.TemporaryDirectory() as tmp:
        sh = Shard(0, 4, "flat", tmp)
        sh._raw_fh = open(Path(tmp) / "x.raw", "wb")
        sh.add(np.ones((2, 4), np.float32), np.array([3, 5]))
        try:
            sh.add(np.ones((1, 4), np.float32), np.array([4]))
        except ValueError:
            print("ok  shard rejects non-ascending ids")
            return
        raise AssertionError("expected ValueError")


def test_flat_backend_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        generate(str(tmp / "v.npy"), N, DIM, clusters=64, seed=0)
        cfg = IndexConfig(num_shards=4, backend="flat", add_batch=3_000)
        build_index(str(tmp / "v.npy"), str(tmp / "idx"), cfg, log=lambda *_: None)

        data = np.load(tmp / "v.npy", mmap_mode="r")
        idx = ShardedIndex.load(tmp / "idx")
        assert idx.n_vectors == N and sum(s.n for s in idx.shards.values()) == N
        for sid, sh in idx.shards.items():  # hash routing: id % num_shards == shard id
            assert (np.asarray(sh.ids) % 4 == sid).all()

        queries = make_queries(data, 50)
        truth = exact_topk(queries, data, K)[1]
        scores, ids = idx.search(queries, K)
        assert _recall(ids, truth) >= 0.99, _recall(ids, truth)  # exact search; fp16 rounding may flip ties
        assert (np.diff(scores, axis=1) <= 1e-6).all(), "scores must be sorted descending"

        # a two-node deployment (shards {0,1} + {2,3}) merged by score must equal the single-node answer
        a = ShardedIndex.load(tmp / "idx", shard_ids=[0, 1])
        b = ShardedIndex.load(tmp / "idx", shard_ids=[2, 3])
        sa, ia = a.search(queries, K)
        sb, ib = b.search(queries, K)
        ms, mi = topk_pad(np.concatenate([sa, sb], 1), np.concatenate([ia, ib], 1), K)
        assert _recall(mi, ids) >= 0.999
        print("ok  flat backend: shard routing, recall, 2-node merge == 1-node")

        # real-time buffer: a brand-new vector is the nearest neighbour of itself immediately
        new_vec = queries[:1].copy()
        idx.upsert(np.array([N + 5]), new_vec)
        s, i = idx.search(new_vec, K)
        assert i[0, 0] == N + 5 and s[0, 0] > 0.999
        try:
            idx.upsert(np.array([7]), new_vec)
            raise AssertionError("upsert of a built id must be rejected")
        except ValueError:
            pass
        idx.persist_delta()
        again = ShardedIndex.load(tmp / "idx")
        assert again.search(new_vec, 1)[1][0, 0] == N + 5
        print("ok  real-time upsert + persistence")

        meta = MetadataStore(tmp / "meta.db")
        meta.put_many([(1, {"title": "a"}), (2, {"title": "b"})])
        assert meta.get_many([1, 2, 3, -1]) == {1: {"title": "a"}, 2: {"title": "b"}}
        print("ok  metadata store")


def test_tiny_index_pads_results():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        generate(str(tmp / "v.npy"), 100, 16, clusters=4)
        build_index(str(tmp / "v.npy"), str(tmp / "idx"), IndexConfig(num_shards=4, backend="flat"), log=lambda *_: None)
        s, i = ShardedIndex.load(tmp / "idx").search(np.ones((1, 16), np.float32), 200)
        assert i.shape == (1, 200) and (i[0] >= 0).sum() == 100 and (i[0, 100:] == -1).all()
        print("ok  k > corpus size is padded with -1")


def test_faiss_backend():
    if not HAVE_FAISS:
        print("SKIP faiss backend (pip install faiss-cpu to run IVF-PQ tests)")
        return
    n, dim = 30_000, 64
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        generate(str(tmp / "v.npy"), n, dim, clusters=128, seed=1)
        cfg = IndexConfig(num_shards=2, backend="faiss", pq_m=8, train_size=15_000, add_batch=5_000, default_nprobe=16)
        build_index(str(tmp / "v.npy"), str(tmp / "idx"), cfg, log=lambda *_: None)
        data = np.load(tmp / "v.npy", mmap_mode="r")
        idx = ShardedIndex.load(tmp / "idx", omp_threads=1)
        queries = make_queries(data, 100)
        truth = exact_topk(queries, data, K)[1]
        r_pq = _recall(idx.search(queries, K, nprobe=16, rerank_factor=0)[1], truth)
        r_rr = _recall(idx.search(queries, K, nprobe=16, rerank_factor=8)[1], truth)
        print(f"    faiss IVF-PQ recall@{K}: PQ-only={r_pq:.3f}  with exact rerank={r_rr:.3f}")
        assert r_rr >= r_pq and r_rr >= 0.85, (r_pq, r_rr)
        print("ok  faiss IVF-PQ backend")


if __name__ == "__main__":
    test_exact_and_pad()
    test_shard_requires_ascending_ids()
    test_flat_backend_end_to_end()
    test_tiny_index_pads_results()
    test_faiss_backend()
    print("\nALL PASSED")
