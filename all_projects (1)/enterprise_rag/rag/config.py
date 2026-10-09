from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").lower() in {"1", "true", "yes"}


@dataclass
class Config:
    """All tunables. Every field can be overridden by constructor argument or RAG_* env var."""

    index_dir: str = field(default_factory=lambda: _env("RAG_INDEX_DIR", "./index"))

    # --- embeddings -------------------------------------------------------
    # "sentence-transformers" for real use, "hash" for fast offline tests.
    embed_backend: str = field(default_factory=lambda: _env("RAG_EMBED_BACKEND", "sentence-transformers"))
    embed_model: str = field(default_factory=lambda: _env("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    embed_batch_size: int = 64
    # bge models expect this prefix on *queries* only; set to "" for models that don't.
    query_instruction: str = field(
        default_factory=lambda: _env(
            "RAG_QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: "
        )
    )

    # --- chunking / ingestion --------------------------------------------
    chunk_size: int = 900  # characters
    chunk_overlap: int = 150
    min_chunk_chars: int = 40
    ingest_workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))

    # --- dense index (HNSW) ----------------------------------------------
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128

    # --- retrieval --------------------------------------------------------
    dense_k: int = 50
    sparse_k: int = 50
    rrf_k: int = 60
    rerank_top_n: int = 30
    final_k: int = 5
    use_reranker: bool = field(default_factory=lambda: _env_bool("RAG_USE_RERANKER", True))
    reranker_model: str = field(default_factory=lambda: _env("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base"))

    # --- generation -------------------------------------------------------
    # "anthropic" (needs ANTHROPIC_API_KEY) or "extractive" (no LLM, offline).
    llm_backend: str = field(default_factory=lambda: _env("RAG_LLM_BACKEND", "anthropic"))
    llm_model: str = field(default_factory=lambda: _env("RAG_LLM_MODEL", "claude-sonnet-5-5"))
    max_tokens: int = 700
    max_context_chars: int = 12000
