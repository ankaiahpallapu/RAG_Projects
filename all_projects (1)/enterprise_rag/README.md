# Enterprise RAG System for Document Intelligence

Production-style Retrieval-Augmented Generation over large document collections (PDF, DOCX, HTML, MD, TXT).

```
 files ──► parallel parse ──► chunk ──┬─► SQLite chunk store
                                      ├─► BM25 sparse index (scipy CSC)
                                      └─► embeddings ─► HNSW dense index (FAISS)

 question ─► dense top-50 ─┐
          └► BM25  top-50 ─┴► Reciprocal Rank Fusion ─► cross-encoder rerank (top-30)
                                                     ─► top-5 passages ─► LLM (cited answer)
```

**Why this design**
- **Hybrid retrieval**: dense vectors catch paraphrases; BM25 catches exact IDs, names, codes, and rare terms. RRF fuses them without needing to calibrate scores.
- **Cross-encoder reranking** of only ~30 candidates is where most of the precision gain comes from, at a cost you can afford per query.
- **Bounded-memory build**: documents stream through in batches, chunk text lives in SQLite, so ingestion RAM doesn't grow with corpus size.
- **Grounded generation**: the LLM only sees numbered passages and must cite them; with no relevant passages it says so.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10-3.12
pip install -r requirements.txt

# 1) Offline smoke test (no downloads, no API key)
python tests/smoke_test.py

# 2) Try it on a synthetic corpus
python -m rag make-sample --out data/sample/docs --docs 500
python -m rag build data/sample/docs --index-dir index --embed-backend hash --no-rerank
python -m rag eval data/sample/eval.jsonl --index-dir index --embed-backend hash --no-rerank

# 3) Real use: your documents, real models
export ANTHROPIC_API_KEY=sk-ant-...
python -m rag build /path/to/your/docs --index-dir index          # downloads bge-small + reranker on first run
python -m rag query "What is our travel reimbursement limit?" --index-dir index
python -m rag serve --index-dir index --port 8000
```

API:
```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
     -d '{"question": "What is the notice period in the vendor contract?", "top_k": 5}'
# /search (retrieval only), /query/stream (SSE), /health
```

## Measuring precision (do this on *your* data)

Create a labelled set, one JSON object per line:
```json
{"question": "What is the notice period in the Acme contract?", "relevant_doc_ids": ["contracts/acme_msa.pdf"]}
```
Then run `python -m rag eval my_eval.jsonl --index-dir index`. You get a comparison table of dense-only, BM25-only, hybrid, and hybrid + rerank with P@k, hit@k, MRR and latency.

- `doc_id` is the file path relative to the folder you ingested.
- **Precision@k** here = fraction of the k returned chunks that come from a relevant document. Quote the exact `k` when you report a number (e.g. "94% precision@5"), and build the eval set from real user questions (100-300 is a good start).
- The synthetic corpus from `make-sample` only validates the plumbing. Its numbers say nothing about real-world quality.

## Scaling notes (100K+ documents)

| Concern | What to do |
|---|---|
| Embedding throughput | Run the build on a GPU box (`sentence-transformers` picks CUDA automatically); raise `embed_batch_size`. |
| Parsing throughput | `--workers N` parses files in parallel processes. |
| RAM | 100K docs at ~10 chunks each is ~1M chunks: about 1.5 GB of vectors for a 384-dim model plus HNSW links. Beyond ~10M chunks switch the dense layer to IVF-PQ (see the `vector_search` project). |
| Latency | Dense + BM25 search is a few ms; the reranker (~30 pairs) and the LLM dominate. Lower `rerank_top_n` or use a smaller reranker for tighter budgets. |
| Freshness | `build` is a full rebuild. For incremental updates keep per-batch indexes and merge, or move to a managed vector DB. |
| Multilingual | Swap `RAG_EMBED_MODEL` for `BAAI/bge-m3` or `intfloat/multilingual-e5-base` (set `RAG_QUERY_INSTRUCTION=""` for e5 with a "query: " convention as needed). |

## Configuration (env vars)

`RAG_INDEX_DIR`, `RAG_EMBED_BACKEND` (`sentence-transformers`|`hash`), `RAG_EMBED_MODEL`, `RAG_USE_RERANKER`, `RAG_RERANKER_MODEL`, `RAG_LLM_BACKEND` (`anthropic`|`extractive`), `RAG_LLM_MODEL` (default `claude-sonnet-5-5`). Chunking and retrieval knobs live in `rag/config.py`.

## Layout

```
rag/ingest.py      file loaders + paragraph/sentence-aware chunker + parallel ingestion
rag/bm25.py        BM25 on sparse matrices (no external search engine)
rag/dense.py       FAISS HNSW (NumPy fallback)
rag/pipeline.py    build / load / hybrid retrieve / answer
rag/reranker.py    cross-encoder reranker
rag/generator.py   Anthropic generator (+ offline extractive fallback)
rag/evaluation.py  precision@k / recall@k / MRR comparison
rag/api.py         FastAPI service
```
