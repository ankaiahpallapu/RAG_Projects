from __future__ import annotations

import dataclasses
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from .bm25 import BM25Index
from .config import Config
from .dense import DenseIndex
from .embeddings import make_embedder
from .generator import make_generator
from .ingest import iter_chunks
from .reranker import make_reranker
from .schema import Chunk, Hit
from .store import ChunkStore

log = logging.getLogger(__name__)
MANIFEST = "manifest.json"
NO_ANSWER = "I could not find an answer in the provided documents."


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankers of 1 / (k + rank). Needs no score calibration."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


class RAGPipeline:
    """Hybrid (dense + BM25) retrieval -> RRF fusion -> cross-encoder rerank -> grounded generation."""

    def __init__(self, cfg: Config, embedder, dense: DenseIndex, bm25: BM25Index, store: ChunkStore):
        self.cfg, self.embedder, self.dense, self.bm25, self.store = cfg, embedder, dense, bm25, store
        self._reranker = None
        self._reranker_ready = False
        self._generator = None

    # ---- lazily constructed heavy components --------------------------------
    @property
    def reranker(self):
        if not self._reranker_ready:
            self._reranker = make_reranker(self.cfg)
            self._reranker_ready = True
        return self._reranker

    @property
    def generator(self):
        if self._generator is None:
            self._generator = make_generator(self.cfg)
        return self._generator

    # ---- build / load ---------------------------------------------------------
    @classmethod
    def build(
        cls,
        data_dir: str | Path,
        cfg: Config | None = None,
        batch_size: int = 512,
        progress: Callable[[int], None] | None = None,
    ) -> "RAGPipeline":
        """Stream documents -> chunks -> (SQLite store, BM25, dense index) in bounded-memory batches."""
        cfg = cfg or Config()
        out = Path(cfg.index_dir)
        out.mkdir(parents=True, exist_ok=True)
        db = out / "chunks.db"
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)

        embedder = make_embedder(cfg)
        dense = DenseIndex(
            embedder.dim, hnsw_m=cfg.hnsw_m, ef_construction=cfg.hnsw_ef_construction, ef_search=cfg.hnsw_ef_search
        )
        bm25 = BM25Index()
        store = ChunkStore(db)

        batch: list[Chunk] = []
        n = 0
        t0 = time.time()

        def flush() -> None:
            if not batch:
                return
            texts = [c.text for c in batch]
            store.add_many(batch)
            bm25.add_documents(texts)
            dense.add(embedder.encode_documents(texts))
            batch.clear()

        for doc_id, pos, text in iter_chunks(data_dir, cfg):
            batch.append(Chunk(n, doc_id, doc_id, pos, text))
            n += 1
            if len(batch) >= batch_size:
                flush()
                if progress:
                    progress(n)
        flush()
        if n == 0:
            raise ValueError(f"no ingestible documents found under {data_dir}")

        bm25.finalize()
        bm25.save(out)
        dense.save(out)
        manifest = {
            "version": 1,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "n_chunks": n,
            "n_docs": store.count_docs(),
            "embed_backend": cfg.embed_backend,
            "embed_model": cfg.embed_model,
            "dim": embedder.dim,
            "dense_backend": dense.backend,
            "build_seconds": round(time.time() - t0, 1),
        }
        (out / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log.info("indexed %d chunks from %d docs in %.1fs", n, manifest["n_docs"], manifest["build_seconds"])
        return cls(cfg, embedder, dense, bm25, store)

    @classmethod
    def load(cls, cfg: Config | None = None) -> "RAGPipeline":
        cfg = cfg or Config()
        out = Path(cfg.index_dir)
        manifest = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
        # Queries must be embedded with exactly the model the index was built with.
        cfg = dataclasses.replace(cfg, embed_backend=manifest["embed_backend"], embed_model=manifest["embed_model"])
        embedder = make_embedder(cfg)
        dense = DenseIndex.load(out, manifest["dense_backend"], manifest["dim"], cfg.hnsw_ef_search)
        bm25 = BM25Index.load(out)
        store = ChunkStore(out / "chunks.db")
        return cls(cfg, embedder, dense, bm25, store)

    # ---- retrieval --------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        final_k: int | None = None,
        mode: str = "hybrid",
        rerank: bool | None = None,
    ) -> list[Hit]:
        """mode: 'hybrid' | 'dense' | 'sparse'.  rerank: None = use reranker if configured."""
        cfg = self.cfg
        final_k = final_k or cfg.final_k
        use_rerank = (rerank is None or rerank) and self.reranker is not None

        dense_ids: list[int] = []
        sparse_ids: list[int] = []
        if mode in ("hybrid", "dense"):
            _, ids = self.dense.search(self.embedder.encode_queries([query]), cfg.dense_k)
            dense_ids = [int(i) for i in ids[0] if i >= 0]
        if mode in ("hybrid", "sparse"):
            ids, _ = self.bm25.search(query, cfg.sparse_k)
            sparse_ids = [int(i) for i in ids]
        if not dense_ids and not sparse_ids:
            return []

        dense_rank = {d: r for r, d in enumerate(dense_ids, 1)}
        sparse_rank = {d: r for r, d in enumerate(sparse_ids, 1)}
        fused = rrf_fuse([r for r in (dense_ids, sparse_ids) if r], cfg.rrf_k)
        ordered = sorted(fused, key=fused.__getitem__, reverse=True)

        n_candidates = max(cfg.rerank_top_n, final_k) if use_rerank else final_k
        cand_ids = ordered[:n_candidates]
        chunks = self.store.get_many(cand_ids)
        hits = [Hit(chunks[i], fused[i], dense_rank.get(i), sparse_rank.get(i)) for i in cand_ids if i in chunks]

        if use_rerank and hits:
            scores = self.reranker.score(query, [h.chunk.text for h in hits])
            for h, s in zip(hits, scores):
                h.rerank_score = float(s)
            hits.sort(key=lambda h: h.rerank_score, reverse=True)
        return hits[:final_k]

    # ---- generation -------------------------------------------------------------
    def answer(self, query: str, final_k: int | None = None, generate: bool = True) -> dict:
        t0 = time.perf_counter()
        hits = self.retrieve(query, final_k)
        t1 = time.perf_counter()
        text = None
        if generate:
            text = self.generator.generate(query, hits) if hits else NO_ANSWER
        t2 = time.perf_counter()
        return {
            "question": query,
            "answer": text,
            "sources": [h.to_dict() for h in hits],
            "timings_ms": {"retrieval": round((t1 - t0) * 1000, 1), "generation": round((t2 - t1) * 1000, 1)},
        }

    def answer_stream(self, query: str, final_k: int | None = None) -> tuple[list[Hit], Iterator[str]]:
        hits = self.retrieve(query, final_k)
        if not hits:
            return hits, iter([NO_ANSWER])
        return hits, self.generator.stream(query, hits)
