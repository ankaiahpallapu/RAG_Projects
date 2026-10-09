"""Sharded IVF-PQ vector search with exact re-ranking, built for 50M+ embeddings and sub-100 ms queries."""

from .cluster import ShardedIndex
from .config import IndexConfig

__all__ = ["ShardedIndex", "IndexConfig"]
