from __future__ import annotations

import json
import re
from array import array
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOP = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their then there "
    "these they this to was were will with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Index:
    """Okapi BM25 over a CSC sparse matrix of precomputed term weights.

    Weights are computed once in `finalize()`, so a query is just a few column slices + adds.
    Documents are addressed by their insertion order (0..N-1), which matches the dense index ids.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.n_docs = 0
        self.mat: sparse.csc_matrix | None = None
        self._rows, self._cols, self._tfs = array("i"), array("i"), array("f")
        self._doc_len = array("i")

    # ---- build -------------------------------------------------------------
    def add_documents(self, texts: Iterable[str]) -> None:
        for text in texts:
            toks = tokenize(text)
            d = self.n_docs
            self.n_docs += 1
            self._doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                tid = self.vocab.setdefault(term, len(self.vocab))
                self._rows.append(d)
                self._cols.append(tid)
                self._tfs.append(float(tf))

    def finalize(self) -> None:
        n, v = self.n_docs, len(self.vocab)
        rows = np.asarray(self._rows, dtype=np.int32)
        cols = np.asarray(self._cols, dtype=np.int32)
        tf = np.asarray(self._tfs, dtype=np.float32)
        dl = np.asarray(self._doc_len, dtype=np.float32)
        avgdl = float(dl.mean()) if n else 1.0
        df = np.bincount(cols, minlength=v).astype(np.float32)
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        denom = tf + self.k1 * (1.0 - self.b + self.b * dl[rows] / max(avgdl, 1e-9))
        weights = idf[cols] * tf * (self.k1 + 1.0) / denom
        self.mat = sparse.csc_matrix((weights, (rows, cols)), shape=(n, v), dtype=np.float32)
        self._rows, self._cols, self._tfs, self._doc_len = array("i"), array("i"), array("f"), array("i")

    # ---- query -------------------------------------------------------------
    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (doc_ids, scores) for the top-k documents, best first. Zero-score docs are dropped."""
        assert self.mat is not None, "call finalize() or load() first"
        terms = [self.vocab[t] for t in set(tokenize(query)) if t in self.vocab]
        if not terms or self.n_docs == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        scores = np.zeros(self.n_docs, dtype=np.float32)
        indptr, indices, data = self.mat.indptr, self.mat.indices, self.mat.data
        for tid in terms:
            s, e = indptr[tid], indptr[tid + 1]
            scores[indices[s:e]] += data[s:e]
        k = min(k, self.n_docs)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        top = top[scores[top] > 0]
        return top.astype(np.int64), scores[top]

    # ---- persistence -------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        sparse.save_npz(d / "bm25.npz", self.mat)
        (d / "bm25_vocab.json").write_text(
            json.dumps({"k1": self.k1, "b": self.b, "n_docs": self.n_docs, "vocab": self.vocab}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "BM25Index":
        d = Path(directory)
        meta = json.loads((d / "bm25_vocab.json").read_text(encoding="utf-8"))
        idx = cls(meta["k1"], meta["b"])
        idx.vocab, idx.n_docs = meta["vocab"], meta["n_docs"]
        idx.mat = sparse.load_npz(d / "bm25.npz").tocsc()
        return idx
