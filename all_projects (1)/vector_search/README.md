# Real-time Vector Search Infrastructure

Sharded **IVF-PQ** vector search with exact re-ranking, designed for 50M+ embeddings and sub-100 ms queries on commodity hardware.

```
                       ┌──────────── gateway (optional, scatter-gather) ────────────┐
   query vector ──────►│  fan out to nodes ─► merge top-k by score (degrades if a node is down)
                       └───────────┬───────────────────────────┬────────────────────┘
                                   ▼                           ▼
                       node A: shards 0,1,2,3        node B: shards 4,5,6,7
                       ┌─ per shard (parallel threads) ──────────────────────────────┐
                       │ HNSW over IVF centroids ─► probe nprobe cells ─► PQ scan    │
                       │ ─► k*rerank_factor candidates ─► exact re-score from fp16   │
                       │    originals on NVMe ─► shard top-k                         │
                       └──────────────────────────────────────────────────────────────┘
                       + in-memory delta buffer (exact) for vectors added since the build
```

## Why it hits the targets

| Technique | Effect |
|---|---|
| **Product quantization** (48 B/vector for 384-d) | The searchable index is ~2.8 GB for 50M vectors instead of ~77 GB of float32. |
| **IVF** (inverted file) | Only `nprobe` of ~8K cells per shard are scanned, never the whole corpus. |
| **HNSW over centroids** | Picking cells is itself sub-linear (IVF8192_HNSW32). |
| **Memory-mapped shards** | Fast start, low RSS, pages are shared via the OS cache. |
| **Exact re-ranking** | PQ is lossy; re-scoring `k*4` candidates against the fp16 originals recovers most of the lost recall for ~40 random 768 B reads per shard. |
| **Hash sharding + parallel fan-out** | Latency tracks one shard (1/8 of the data); FAISS releases the GIL, so threads scale. Add nodes behind the gateway to scale further. |
| **Delta buffer** | New vectors are searchable immediately (exact search over a small in-memory buffer) with no index rebuild. |

Sizing for **50M x 384-d**: PQ codes + ids = 50M x 56 B = **2.8 GB RAM/page cache**; fp16 originals for re-rank = **38 GB on NVMe** (not RAM); the id arrays add 0.4 GB.
`--no-raw` drops the originals if you'd rather trade recall for disk.

## Quick start

```bash
pip install -r requirements.txt          # Python 3.10-3.12

python tests/smoke_test.py               # runs offline; faiss tests run only if faiss-cpu is installed

# 1M-vector demo (synthetic embeddings, ~0.8 GB)
python -m vsearch.datagen --out data/vectors.npy --n 1000000 --dim 384
python -m vsearch.build   --data data/vectors.npy --out index --shards 8
python -m vsearch.benchmark --index index --data data/vectors.npy --queries 200 --truth-cache data/truth.npy

# serve
VS_INDEX_DIR=index uvicorn vsearch.api:app --host 0.0.0.0 --port 8000 --workers 1
curl -s localhost:8000/search -H 'content-type: application/json' \
     -d "{\"vector\": $(python -c 'import random,json;print(json.dumps([random.gauss(0,1) for _ in range(384)]))'), \"k\": 5}"
```

**Your own embeddings:** save them as a float16/float32 `.npy` of shape `(N, dim)`. Row number = vector id. Keep payloads (titles, URLs...) in `MetadataStore` (SQLite), keyed by that id (`vsearch.metadata.load_jsonl`), and pass `"with_metadata": true` on search.

## The 50M recipe

```bash
python -m vsearch.datagen --out data/50m.npy --n 50000000 --dim 384        # 38 GB, or use your real embeddings
python -m vsearch.build --data data/50m.npy --out index50m --shards 8 --train-size 500000
python -m vsearch.benchmark --index index50m --data data/50m.npy --queries 100 --truth-cache data/truth50m.npy
```
Build time is dominated by k-means training (~minutes per shard on a many-core box). To parallelise across machines with shared storage, run `build --only 0,1` on one machine, `--only 2,3` on the next, and so on. Shards are independent.
Exact ground truth over 50M vectors takes a few minutes of brute-force matmul; `--truth-cache` stores it.

## Tuning (use the benchmark on a sample of YOUR embeddings)

`benchmark` sweeps `nprobe` x `rerank_factor`, prints recall@k against exact search plus p50/p95/p99 latency, picks the best-recall point under your p99 budget (`--target-ms`), then load-tests it at several concurrency levels.

| Knob | Raise it to... | Costs |
|---|---|---|
| `nprobe` | gain recall | latency (linear in cells scanned) |
| `rerank_factor` | recover PQ error | a few random reads per shard |
| `pq_m` | gain PQ accuracy (more bytes/vector) | RAM, scan time |
| `nlist` | cut scan cost per cell | training time, needs more training vectors |
| `--shards` / nodes | cut latency, raise capacity | fan-out overhead |

Real embeddings behave differently from the synthetic mixture, so **do not quote latency or recall numbers until you've run the benchmark on your own data and hardware.** The sub-100 ms target assumes NVMe storage and enough cores for the fan-out.

## Deploy

```bash
docker build -t vsearch .
docker run -p 8000:8000 -v $PWD/index:/data/index:ro vsearch                     # one node, all shards
docker run -e VS_SHARDS=0,1,2,3 ... vsearch                                      # node A
docker run -e VS_SHARDS=4,5,6,7 ... vsearch                                      # node B
VS_NODES=http://node-a:8000,http://node-b:8000 uvicorn vsearch.gateway:app       # gateway
```
Endpoints: `POST /search`, `/batch_search`, `/upsert`, `/persist`; `GET /stats` (live p50/p95/p99), `/health`. Set `VS_EMBED_MODEL` to let `/search` accept raw `text`.

## Real-time updates and limitations

- `POST /upsert` puts new vectors (ids >= the built range) into an exact in-memory buffer that is searched alongside the shards, so they are visible instantly. `POST /persist` writes the buffer to disk so it survives a restart. Keep it under ~100K vectors, then fold it in with a rebuild.
- Built shards are immutable: no in-place updates or deletes of existing ids (rebuild, or filter tombstones in your application).
- Upserts are a single-node feature; the gateway only does reads.
- Vectors are L2-normalised on the way in, so scores are cosine similarity.

## Layout

```
vsearch/shard.py     one partition: IVF-HNSW,PQ index + fp16 raw store + exact re-rank
vsearch/cluster.py   ShardedIndex (parallel fan-out + merge) and the real-time DeltaBuffer
vsearch/build.py     two-phase build: train per shard, then a single pass over the data
vsearch/benchmark.py recall + latency + concurrency measurement vs exact ground truth
vsearch/api.py       FastAPI search node        vsearch/gateway.py   scatter-gather gateway
vsearch/datagen.py   synthetic embeddings       vsearch/metadata.py  id -> payload store
```
