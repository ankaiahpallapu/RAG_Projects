from __future__ import annotations

import re
import zlib
from typing import Sequence

import numpy as np

from .config import Config


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64, query_instruction: str = "", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.dim = int(self.model.encode(["dimension probe"], convert_to_numpy=True).shape[1])

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32, copy=False)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.query_instruction + t for t in texts])


class HashEmbedder:
    """Deterministic signed feature-hashing embedder (unigrams + bigrams).

    Not semantic - it exists so the whole pipeline can be exercised offline in tests/CI
    without downloading a model.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = re.findall(r"\w+", text.lower())
            grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for g in grams:
                h = zlib.crc32(g.encode("utf-8"))
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    encode_documents = _encode
    encode_queries = _encode


def make_embedder(cfg: Config):
    if cfg.embed_backend == "hash":
        return HashEmbedder()
    if cfg.embed_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(
            cfg.embed_model, batch_size=cfg.embed_batch_size, query_instruction=cfg.query_instruction
        )
    raise ValueError(f"unknown embed_backend: {cfg.embed_backend!r}")
