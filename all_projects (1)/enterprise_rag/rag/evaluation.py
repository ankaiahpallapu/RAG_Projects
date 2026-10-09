"""Retrieval evaluation.

Eval set: JSONL, one object per line:
    {"question": "...", "relevant_doc_ids": ["policies/travel.pdf"]}
`doc_id` is the file path relative to the ingested root.

Metrics (all averaged over questions):
  precision@k : fraction of the k returned chunks that come from a relevant document
                (empty slots count as misses, so a short result list can't inflate it)
  hit@k       : 1 if any of the top-k chunks is from a relevant document
  recall@k    : fraction of the relevant documents that appear in the top-k
  MRR         : 1 / rank of the first relevant chunk
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .pipeline import RAGPipeline


def load_eval_set(path: str | Path) -> list[dict]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if not item.get("relevant_doc_ids"):
                raise ValueError(f"eval item without relevant_doc_ids: {item}")
            items.append(item)
    return items


def run_eval(
    pipeline: RAGPipeline,
    items: list[dict],
    ks: tuple[int, ...] = (1, 3, 5, 10),
    mode: str = "hybrid",
    rerank: bool | None = None,
) -> dict:
    kmax = max(ks)
    prec = {k: [] for k in ks}
    hit = {k: [] for k in ks}
    rec = {k: [] for k in ks}
    rr: list[float] = []
    lat: list[float] = []
    for it in items:
        rel = set(it["relevant_doc_ids"])
        t0 = time.perf_counter()
        hits = pipeline.retrieve(it["question"], final_k=kmax, mode=mode, rerank=rerank)
        lat.append((time.perf_counter() - t0) * 1000)
        docs = [h.chunk.doc_id for h in hits]
        for k in ks:
            top = docs[:k]
            prec[k].append(sum(d in rel for d in top) / k)
            hit[k].append(float(any(d in rel for d in top)))
            rec[k].append(len(rel & set(top)) / len(rel))
        first = next((i for i, d in enumerate(docs, 1) if d in rel), None)
        rr.append(1.0 / first if first else 0.0)
    return {
        "n_questions": len(items),
        "precision": {k: float(np.mean(v)) for k, v in prec.items()},
        "hit": {k: float(np.mean(v)) for k, v in hit.items()},
        "recall": {k: float(np.mean(v)) for k, v in rec.items()},
        "mrr": float(np.mean(rr)),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
    }


def compare_modes(pipeline: RAGPipeline, items: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, dict]:
    configs = [("dense", "dense", False), ("bm25", "sparse", False), ("hybrid (RRF)", "hybrid", False)]
    if pipeline.reranker is not None:
        configs.append(("hybrid + rerank", "hybrid", True))
    return {name: run_eval(pipeline, items, ks, mode, rerank) for name, mode, rerank in configs}


def format_table(results: dict[str, dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> str:
    head = f"{'method':<18}" + "".join(f"{'P@' + str(k):>8}" for k in ks) + f"{'hit@5':>8}{'MRR':>8}{'p50 ms':>9}{'p95 ms':>9}"
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        h5 = r["hit"].get(5, r["hit"][max(ks)])
        lines.append(
            f"{name:<18}"
            + "".join(f"{r['precision'][k]:>8.3f}" for k in ks)
            + f"{h5:>8.3f}{r['mrr']:>8.3f}{r['latency_ms_p50']:>9.1f}{r['latency_ms_p95']:>9.1f}"
        )
    return "\n".join(lines)
