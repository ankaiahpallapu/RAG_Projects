from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalise rows (float32 copy). On unit vectors, inner product == cosine similarity."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def topk_pad(scores: np.ndarray, ids: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Sort rows by descending score, keep k columns, and pad with (-inf, -1) if fewer than k exist."""
    order = np.argsort(-scores, axis=1)[:, :k]
    scores = np.take_along_axis(scores, order, axis=1)
    ids = np.take_along_axis(ids, order, axis=1)
    short = k - scores.shape[1]
    if short > 0:
        n = scores.shape[0]
        scores = np.concatenate([scores, np.full((n, short), -np.inf, np.float32)], axis=1)
        ids = np.concatenate([ids, np.full((n, short), -1, np.int64)], axis=1)
    return scores, ids


def exact_topk(
    queries: np.ndarray,
    vectors,
    k: int,
    ids: np.ndarray | None = None,
    block: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact inner-product top-k over a (possibly memory-mapped, float16) matrix, in blocks.

    Returns (scores, ids) of shape (nq, k). `ids` maps row -> external id (default: row number).
    """
    q = np.ascontiguousarray(queries, dtype=np.float32)
    nq, n = len(q), len(vectors)
    keep = max(1, min(k, n))
    best_s = np.empty((nq, 0), np.float32)
    best_i = np.empty((nq, 0), np.int64)
    for start in range(0, n, block):
        blk = np.asarray(vectors[start : start + block], dtype=np.float32)
        sims = q @ blk.T
        stop = start + len(blk)
        blk_ids = np.arange(start, stop, dtype=np.int64) if ids is None else np.asarray(ids[start:stop], np.int64)
        cand_s = np.concatenate([best_s, sims], axis=1)
        cand_i = np.concatenate([best_i, np.broadcast_to(blk_ids, (nq, len(blk_ids)))], axis=1)
        if cand_s.shape[1] > keep:
            top = np.argpartition(-cand_s, keep - 1, axis=1)[:, :keep]
            cand_s = np.take_along_axis(cand_s, top, axis=1)
            cand_i = np.take_along_axis(cand_i, top, axis=1)
        best_s, best_i = cand_s, cand_i
    return topk_pad(best_s, best_i, k)
