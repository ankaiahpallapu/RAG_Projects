from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Iterable

from .schema import Chunk


class ChunkStore:
    """SQLite-backed chunk store: keeps RAM flat no matter how many chunks are indexed."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                   chunk_id INTEGER PRIMARY KEY,
                   doc_id   TEXT NOT NULL,
                   source   TEXT NOT NULL,
                   position INTEGER NOT NULL,
                   text     TEXT NOT NULL)"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
        self._conn.commit()

    def add_many(self, chunks: Iterable[Chunk]) -> None:
        rows = [(c.chunk_id, c.doc_id, c.source, c.position, c.text) for c in chunks]
        with self._lock:
            self._conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)", rows)
            self._conn.commit()

    def get_many(self, ids: list[int]) -> dict[int, Chunk]:
        out: dict[int, Chunk] = {}
        ids = [int(i) for i in ids]
        with self._lock:
            for start in range(0, len(ids), 500):  # stay under SQLite's variable limit
                part = ids[start : start + 500]
                q = ",".join("?" * len(part))
                cur = self._conn.execute(
                    f"SELECT chunk_id, doc_id, source, position, text FROM chunks WHERE chunk_id IN ({q})", part
                )
                for row in cur:
                    out[row[0]] = Chunk(*row)
        return out

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def count_docs(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
