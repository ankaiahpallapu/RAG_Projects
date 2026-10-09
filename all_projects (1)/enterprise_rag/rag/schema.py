from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    chunk_id: int
    doc_id: str
    source: str
    position: int
    text: str


@dataclass(slots=True)
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None

    def to_dict(self, include_text: bool = True) -> dict:
        d = {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "source": self.chunk.source,
            "position": self.chunk.position,
            "score": round(float(self.score), 5),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "rerank_score": None if self.rerank_score is None else round(float(self.rerank_score), 5),
        }
        if include_text:
            d["text"] = self.chunk.text
        return d
