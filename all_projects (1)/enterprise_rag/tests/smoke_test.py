"""Offline end-to-end test (no model downloads, no API keys):  python tests/smoke_test.py"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.bm25 import BM25Index  # noqa: E402
from rag.config import Config  # noqa: E402
from rag.evaluation import compare_modes, format_table, load_eval_set  # noqa: E402
from rag.ingest import split_text  # noqa: E402
from rag.pipeline import RAGPipeline, rrf_fuse  # noqa: E402
from rag.sample_data import generate  # noqa: E402


def test_split_text():
    para = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_text(f"Intro paragraph.\n\n{para}\n\nOutro paragraph.", size=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 + 60 + 5 for c in chunks), max(len(c) for c in chunks)
    assert chunks[0].startswith("Intro")
    assert split_text("", 300, 60) == []
    print("ok  split_text:", len(chunks), "chunks")


def test_bm25():
    idx = BM25Index()
    # equal-length docs, so only term frequency differs: doc 1 (tf=2) must beat doc 0 (tf=1)
    idx.add_documents(["alpha beta gamma", "alpha alpha delta", "epsilon zeta eta"])
    idx.finalize()
    ids, scores = idx.search("alpha", 3)
    assert list(ids) == [1, 0], ids  # doc 2 has no match and is dropped
    assert scores[0] > scores[1] > 0
    assert len(idx.search("zebra", 3)[0]) == 0
    # length normalisation: a short doc beats a long one with the same tf
    idx2 = BM25Index()
    idx2.add_documents(["alpha beta", "alpha beta gamma delta epsilon zeta eta theta", "unrelated words here"])
    idx2.finalize()
    assert list(idx2.search("alpha", 2)[0]) == [0, 1]
    print("ok  bm25 ranking")


def test_rrf():
    fused = rrf_fuse([[1, 2, 3], [3, 2, 9]], k=60)
    assert max(fused, key=fused.get) in (2, 3)
    assert set(fused) == {1, 2, 3, 9}
    print("ok  rrf fusion")


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        eval_path = generate(tmp / "docs", n_docs=60, seed=1)
        cfg = Config(
            index_dir=str(tmp / "index"),
            embed_backend="hash",
            llm_backend="extractive",
            use_reranker=False,
            ingest_workers=1,
        )
        pipe = RAGPipeline.build(tmp / "docs", cfg)
        assert pipe.store.count_docs() == 60

        # reload from disk, as the API would
        pipe = RAGPipeline.load(cfg)
        items = load_eval_set(eval_path)
        results = compare_modes(pipe, items, ks=(1, 3, 5))
        print(format_table(results, ks=(1, 3, 5)))
        hybrid = results["hybrid (RRF)"]
        assert hybrid["hit"][5] >= 0.95, hybrid
        assert hybrid["precision"][1] >= 0.8, hybrid

        out = pipe.answer("Who is the owner of project " + items[0]["question"].split("project ")[1])
        assert out["sources"] and out["answer"], out
        assert out["sources"][0]["doc_id"] == items[0]["relevant_doc_ids"][0]
        print("ok  end-to-end:", out["answer"][:100])


if __name__ == "__main__":
    test_split_text()
    test_bm25()
    test_rrf()
    test_end_to_end()
    print("\nALL PASSED")
