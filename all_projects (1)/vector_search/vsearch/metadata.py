from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterable


class MetadataStore:
    """id -> JSON payload lookup (SQLite). Keeps payloads out of the ANN index, which only stores codes."""

    def __init__(self, path: str | Path):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        self._conn.commit()

    def put_many(self, items: Iterable[tuple[int, dict]]) -> None:
        rows = [(int(i), json.dumps(p, separators=(",", ":"))) for i, p in items]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)", rows)
            self._conn.commit()

    def get_many(self, ids: Iterable[int]) -> dict[int, dict]:
        ids = [int(i) for i in ids if i >= 0]
        out: dict[int, dict] = {}
        with self._lock:
            for s in range(0, len(ids), 500):
                part = ids[s : s + 500]
                cur = self._conn.execute(
                    f"SELECT id, payload FROM meta WHERE id IN ({','.join('?' * len(part))})", part
                )
                out.update({row[0]: json.loads(row[1]) for row in cur})
        return out

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0]


def load_jsonl(store: MetadataStore, path: str | Path, id_field: str = "id", batch: int = 50_000) -> int:
    """Bulk-load a JSONL file where each line has an integer id field plus arbitrary payload fields."""
    n, buf = 0, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            buf.append((obj.pop(id_field), obj))
            if len(buf) >= batch:
                store.put_many(buf)
                n += len(buf)
                buf = []
    if buf:
        store.put_many(buf)
        n += len(buf)
    return n
