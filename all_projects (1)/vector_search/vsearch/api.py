"""Search node.   VS_INDEX_DIR=index uvicorn vsearch.api:app --host 0.0.0.0 --port 8000

Env: VS_INDEX_DIR, VS_SHARDS (e.g. "0,1,2" to serve a subset), VS_OMP_THREADS (default 1),
     VS_EMBED_MODEL (optional sentence-transformers model so /search accepts raw `text`).
Run a single uvicorn worker: concurrency comes from threads sharing one memory-mapped index.
"""

from __future__ import annotations

import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .cluster import ShardedIndex
from .metadata import MetadataStore

state: dict = {}
latencies: deque[float] = deque(maxlen=10_000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = os.getenv("VS_INDEX_DIR", "./index")
    shards = os.getenv("VS_SHARDS")
    index = ShardedIndex.load(
        root,
        shard_ids=[int(x) for x in shards.split(",")] if shards else None,
        omp_threads=int(os.getenv("VS_OMP_THREADS", "1")),
    )
    state["index"] = index
    meta_path = Path(root) / "metadata.db"
    state["meta"] = MetadataStore(meta_path) if meta_path.exists() else None
    model = os.getenv("VS_EMBED_MODEL")
    if model:
        from sentence_transformers import SentenceTransformer

        state["embedder"] = SentenceTransformer(model)
    # warm-up: fault in memory-mapped pages and spin up the worker threads
    index.search(np.random.default_rng(0).standard_normal((1, index.cfg.dim)).astype(np.float32), 10)
    yield
    index.close()


app = FastAPI(title="Vector Search Node", version="1.0.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    vector: list[float] | None = None
    text: str | None = None
    k: int = Field(10, ge=1, le=1000)
    nprobe: int | None = Field(None, ge=1, le=4096)
    rerank_factor: int | None = Field(None, ge=0, le=64)
    with_metadata: bool = False


class BatchSearchRequest(BaseModel):
    vectors: list[list[float]] = Field(..., min_length=1, max_length=256)
    k: int = Field(10, ge=1, le=1000)
    nprobe: int | None = Field(None, ge=1, le=4096)
    rerank_factor: int | None = Field(None, ge=0, le=64)


class UpsertRequest(BaseModel):
    ids: list[int] = Field(..., min_length=1, max_length=10_000)
    vectors: list[list[float]]
    metadata: list[dict] | None = None


def _index() -> ShardedIndex:
    index = state.get("index")
    if index is None:
        raise HTTPException(503, "index not loaded")
    return index


def _hits(scores: np.ndarray, ids: np.ndarray, with_metadata: bool) -> list[dict]:
    valid = ids >= 0
    meta = state["meta"].get_many(ids[valid].tolist()) if with_metadata and state.get("meta") else {}
    out = []
    for s, i in zip(scores[valid], ids[valid]):
        hit = {"id": int(i), "score": round(float(s), 5)}
        if with_metadata:
            hit["metadata"] = meta.get(int(i))
        out.append(hit)
    return out


def _search(req: SearchRequest) -> dict:
    index = _index()
    t0 = time.perf_counter()
    if (req.vector is None) == (req.text is None):
        raise HTTPException(422, "provide exactly one of `vector` or `text`")
    if req.text is not None:
        embedder = state.get("embedder")
        if embedder is None:
            raise HTTPException(400, "text search needs VS_EMBED_MODEL to be set on the server")
        q = embedder.encode([req.text], normalize_embeddings=True)
    else:
        q = np.asarray([req.vector], dtype=np.float32)
    if q.shape[1] != index.cfg.dim:
        raise HTTPException(422, f"expected {index.cfg.dim} dimensions, got {q.shape[1]}")
    scores, ids = index.search(q, req.k, req.nprobe, req.rerank_factor)
    took = (time.perf_counter() - t0) * 1000
    latencies.append(took)
    return {"results": _hits(scores[0], ids[0], req.with_metadata), "took_ms": round(took, 2)}


@app.post("/search")
async def search(req: SearchRequest) -> dict:
    return await run_in_threadpool(_search, req)


@app.post("/batch_search")
async def batch_search(req: BatchSearchRequest) -> dict:
    def run() -> dict:
        index = _index()
        q = np.asarray(req.vectors, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != index.cfg.dim:
            raise HTTPException(422, f"expected vectors of dimension {index.cfg.dim}")
        t0 = time.perf_counter()
        scores, ids = index.search(q, req.k, req.nprobe, req.rerank_factor)
        return {
            "results": [_hits(s, i, False) for s, i in zip(scores, ids)],
            "took_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    return await run_in_threadpool(run)


@app.post("/upsert")
async def upsert(req: UpsertRequest) -> dict:
    """Make new vectors searchable immediately (held in an exact in-memory buffer until the next rebuild)."""

    def run() -> dict:
        index = _index()
        vecs = np.asarray(req.vectors, dtype=np.float32)
        if len(vecs) != len(req.ids) or vecs.ndim != 2 or vecs.shape[1] != index.cfg.dim:
            raise HTTPException(422, "ids/vectors length or dimension mismatch")
        try:
            index.upsert(np.asarray(req.ids), vecs)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if req.metadata and state.get("meta"):
            state["meta"].put_many(zip(req.ids, req.metadata))
        return {"upserted": len(req.ids), "delta_vectors": len(index.delta)}

    return await run_in_threadpool(run)


@app.post("/persist")
async def persist() -> dict:
    """Write the real-time buffer to disk so it survives a restart."""
    await run_in_threadpool(_index().persist_delta)
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    lat = np.asarray(latencies) if latencies else np.zeros(1)
    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    return {
        **_index().stats(),
        "recent_queries": len(latencies),
        "latency_ms": {"p50": round(float(p50), 2), "p95": round(float(p95), 2), "p99": round(float(p99), 2)},
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if state.get("index") else "loading"}
