"""FastAPI service.  Run with:  uvicorn rag.api:app --host 0.0.0.0 --port 8000"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipeline"] = RAGPipeline.load()
    yield
    state.clear()


app = FastAPI(title="Enterprise RAG - Document Intelligence", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    include_text: bool = True


def _pipeline() -> RAGPipeline:
    p = state.get("pipeline")
    if p is None:
        raise HTTPException(status_code=503, detail="index not loaded")
    return p


@app.get("/health")
def health() -> dict:
    p = _pipeline()
    return {"status": "ok", "chunks": p.store.count(), "dense_backend": p.dense.backend}


@app.post("/search")
def search(req: QueryRequest) -> dict:
    """Retrieval only (hybrid + rerank), no LLM call."""
    hits = _pipeline().retrieve(req.question, final_k=req.top_k)
    return {"hits": [h.to_dict(req.include_text) for h in hits]}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    """Full RAG: retrieve, rerank, then generate a cited answer."""
    result = _pipeline().answer(req.question, final_k=req.top_k)
    if not req.include_text:
        for s in result["sources"]:
            s.pop("text", None)
    return result


@app.post("/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    """Server-sent events: one `sources` event, many `token` events, then `done`."""
    hits, tokens = _pipeline().answer_stream(req.question, final_k=req.top_k)

    def sse(event: str, data) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def events():
        yield sse("sources", [h.to_dict(req.include_text) for h in hits])
        for tok in tokens:
            yield sse("token", tok)
        yield sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream")
