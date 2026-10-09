from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from .config import Config

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker: scores (query, passage) pairs jointly - slower but far more precise
    than bi-encoder similarity, so it is applied only to the fused top-N candidates."""

    def __init__(self, model_name: str, batch_size: int = 32, max_length: int = 512, device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.batch_size = batch_size

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.empty(0, dtype=np.float32)
        pairs = [(query, p) for p in passages]
        return np.asarray(
            self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False), dtype=np.float32
        )


def make_reranker(cfg: Config) -> CrossEncoderReranker | None:
    if not cfg.use_reranker:
        return None
    try:
        return CrossEncoderReranker(cfg.reranker_model)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully to hybrid-only retrieval
        log.warning("reranker unavailable (%s); continuing without reranking", exc)
        return None
