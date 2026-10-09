# LLM & Search Infrastructure Portfolio

Three production-style projects covering the core of a modern LLM stack: **retrieval-augmented generation**, **multilingual model fine-tuning**, and **large-scale vector search**. Each is a self-contained Python project with its own README, tests, CLI and (where relevant) HTTP API.

![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-portfolio-informational)

| # | Project | What it does | Stack |
|---|---------|--------------|-------|
| 1 | [**Enterprise RAG for Document Intelligence**](enterprise_rag/) | Ingests PDF/DOCX/HTML/MD/TXT at scale and answers questions with cited, grounded responses using hybrid search, reranking and an LLM | FAISS (HNSW), BM25, sentence-transformers, cross-encoder reranker, Claude API, FastAPI, SQLite |
| 2 | [**Multilingual LLM Fine-tuning Pipeline**](multilingual_finetune/) | Config-driven path from raw instruction data to a merged, benchmarked model across 5+ languages | PyTorch, Hugging Face Transformers, PEFT (LoRA / QLoRA), bitsandbytes |
| 3 | [**Real-time Vector Search Infrastructure**](vector_search/) | Sharded IVF-PQ vector database design for tens of millions of embeddings with exact re-ranking and live updates | FAISS (IVF-HNSW-PQ), NumPy, FastAPI, httpx, Docker |

---

## 1. Enterprise RAG for Document Intelligence

```
files ─► parallel parse ─► chunk ─┬─► SQLite chunk store
                                  ├─► BM25 sparse index
                                  └─► embeddings ─► HNSW dense index

question ─► dense top-50 ─┐
         └► BM25  top-50 ─┴► Reciprocal Rank Fusion ─► cross-encoder rerank ─► LLM (cited answer)
```

- **Hybrid retrieval**: dense vectors catch paraphrases, BM25 catches exact IDs, names and rare terms; Reciprocal Rank Fusion merges them without score calibration.
- **Cross-encoder reranking** of the fused top candidates for precision.
- **Bounded-memory ingestion**: documents stream in batches across worker processes, chunk text lives in SQLite.
- **Grounded generation**: the LLM sees only numbered passages, must cite them, and says so when the answer isn't in the documents.
- **Built-in evaluation**: `rag eval` compares dense vs BM25 vs hybrid vs hybrid + rerank on precision@k, hit@k, recall@k and MRR.
- **API**: `/query`, `/search`, `/query/stream` (SSE), `/health`.

```bash
cd enterprise_rag && pip install -r requirements.txt
python tests/smoke_test.py                                    # offline, no downloads
python -m rag build ./my_docs --index-dir index
python -m rag query "What is the notice period in the vendor contract?" --index-dir index
python -m rag eval my_eval.jsonl --index-dir index            # measure precision on YOUR data
python -m rag serve --index-dir index
```

## 2. Multilingual LLM Fine-tuning Pipeline

```
instruction data ─► clean + dedupe ─► language-balanced sampling ─► stratified train/val
     └─► LoRA / QLoRA training (response-only loss) ─► adapter
            ├─► per-language perplexity: base vs tuned
            └─► merge ─► latency benchmark vs unmerged baseline
```

- **6 languages out of the box** (English, Hindi, Spanish, French, German, Telugu); any number via `languages:` in the YAML.
- **Language balancing** with a tunable exponent so high-resource languages don't drown out low-resource ones.
- **Response-only loss** with the same chat template for training and inference.
- **LoRA or 4-bit QLoRA**, gradient checkpointing, auto-resume from checkpoints, multi-GPU via `accelerate`.
- **Serving optimisation**: merge the adapter into a dense checkpoint, run in bf16 with batched generation, benchmark against the unmerged baseline per language.
- Everything is overridable from the CLI: `--set train.epochs=1 --set train.qlora=true`.

```bash
cd multilingual_finetune && pip install -r requirements.txt
python tests/test_data.py && python tests/test_formatting.py
python -m mlft.data_prep --config config.yaml
python -m mlft.train     --config config.yaml
python -m mlft.evaluate  --config config.yaml
python -m mlft.merge     --config config.yaml
python -m mlft.benchmark --config config.yaml --baseline-adapter outputs/run1/adapter
```

## 3. Real-time Vector Search Infrastructure

```
query ─► gateway (optional scatter-gather) ─► nodes ─► shards (parallel)
         per shard: HNSW over IVF centroids ─► probe cells ─► PQ scan
                    ─► k×4 candidates ─► exact re-score from fp16 originals ─► top-k
         + in-memory delta buffer for vectors added since the last build
```

- **Product quantization**: 48 bytes per 384-d vector, so a 50M-vector index needs roughly 3 GB of RAM instead of ~77 GB.
- **IVF + HNSW centroid search**: only a handful of cells are scanned per query.
- **Exact re-ranking** against memory-mapped fp16 originals recovers the recall PQ loses.
- **Hash sharding with parallel fan-out**; a **gateway** merges results across machines and degrades gracefully if a node is down.
- **Real-time upserts** via an exact in-memory delta buffer, searchable immediately.
- **Benchmark tool** sweeps `nprobe × rerank_factor`, reports recall@k versus exact ground truth and p50/p95/p99 latency, and picks the best operating point under your latency budget.

```bash
cd vector_search && pip install -r requirements.txt
python tests/smoke_test.py
python -m vsearch.datagen --out data/vectors.npy --n 1000000 --dim 384
python -m vsearch.build   --data data/vectors.npy --out index --shards 8
python -m vsearch.benchmark --index index --data data/vectors.npy
VS_INDEX_DIR=index uvicorn vsearch.api:app --port 8000
```

---

## Results

Fill this in from your own runs; each number comes from a script in the repo.

| Project | Metric | Result | Measured with | Setup |
|---|---|---|---|---|
| RAG | Precision@5 (hybrid + rerank) | _TBD_ | `python -m rag eval` | _N docs, N labelled questions_ |
| Fine-tuning | Per-request latency reduction | _TBD_ | `python -m mlft.benchmark` | _GPU, baseline vs optimised config_ |
| Fine-tuning | Per-language perplexity change | _TBD_ | `python -m mlft.evaluate` | _model, dataset_ |
| Vector search | Recall@10 / p99 latency | _TBD_ | `python -m vsearch.benchmark` | _N vectors, CPU/NVMe_ |

> The synthetic data generators in each project exist to test the plumbing. Quote metrics only from runs on real data and real hardware.

## Repository layout

```
.
├── enterprise_rag/          # RAG: rag/ (ingest, bm25, dense, reranker, generator, pipeline, api, evaluation), tests/
├── multilingual_finetune/   # Fine-tuning: mlft/ (data_prep, train, evaluate, merge, benchmark), config.yaml, tests/
└── vector_search/           # Vector DB: vsearch/ (shard, cluster, build, benchmark, api, gateway), Dockerfile, tests/
```

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv && source .venv/bin/activate
cd enterprise_rag && pip install -r requirements.txt && python tests/smoke_test.py
```

Each project installs independently, so you only need the dependencies of the one you're running. GPU is required for real fine-tuning; the other two run on CPU.

## Design principles

- **Measure, don't assume**: every project ships an evaluation or benchmark script and reports its own numbers.
- **Offline-testable**: core logic has tests that run without model downloads, API keys or GPUs.
- **Graceful degradation**: optional components (FAISS, reranker, LLM) fall back instead of failing.
- **Honest limits**: each README states what is not covered (for example, built vector shards are immutable and rebuilt offline).

## License

MIT. Add a `LICENSE` file before publishing.
