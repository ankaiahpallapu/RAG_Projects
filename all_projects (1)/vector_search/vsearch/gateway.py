"""Scatter-gather gateway for multi-machine deployments.

Each node serves a subset of shards (VS_SHARDS=0,1 / 2,3 / ...). The gateway fans a query out to every
node concurrently, merges the per-node top-k by score, and degrades gracefully if a node is down or slow.

    VS_NODES=http://node-a:8000,http://node-b:8000 uvicorn vsearch.gateway:app --port 9000

Nodes already re-rank exactly, so scores are comparable across nodes and the merge is exact.
(Real-time /upsert is a single-node feature: send writes to the node that owns the buffer.)
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

NODES = [u.strip().rstrip("/") for u in os.getenv("VS_NODES", "").split(",") if u.strip()]
TIMEOUT_S = float(os.getenv("VS_NODE_TIMEOUT", "0.5"))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not NODES:
        raise RuntimeError("set VS_NODES to a comma-separated list of node URLs")
    state["client"] = httpx.AsyncClient(timeout=TIMEOUT_S, limits=httpx.Limits(max_connections=512))
    yield
    await state["client"].aclose()


app = FastAPI(title="Vector Search Gateway", version="1.0.0", lifespan=lifespan)


class GatewaySearch(BaseModel):
    vector: list[float]
    k: int = Field(10, ge=1, le=1000)
    nprobe: int | None = Field(None, ge=1, le=4096)
    rerank_factor: int | None = Field(None, ge=0, le=64)
    with_metadata: bool = False


async def _call(url: str, payload: dict) -> list[dict] | None:
    try:
        resp = await state["client"].post(f"{url}/search", json=payload)
        resp.raise_for_status()
        return resp.json()["results"]
    except Exception:  # noqa: BLE001 - a failed node means partial results, not a failed query
        return None


@app.post("/search")
async def search(req: GatewaySearch) -> dict:
    payload = req.model_dump(exclude_none=True)
    responses = await asyncio.gather(*[_call(u, payload) for u in NODES])
    ok = [r for r in responses if r is not None]
    if not ok:
        raise HTTPException(503, "no search nodes responded")
    merged = sorted((hit for r in ok for hit in r), key=lambda h: -h["score"])[: req.k]
    return {"results": merged, "degraded": len(ok) < len(NODES), "nodes_ok": len(ok), "nodes_total": len(NODES)}


@app.get("/health")
async def health() -> dict:
    checks = await asyncio.gather(
        *[state["client"].get(f"{u}/health") for u in NODES], return_exceptions=True
    )
    up = sum(1 for c in checks if isinstance(c, httpx.Response) and c.status_code == 200)
    return {"status": "ok" if up == len(NODES) else "degraded", "nodes_up": up, "nodes_total": len(NODES)}
