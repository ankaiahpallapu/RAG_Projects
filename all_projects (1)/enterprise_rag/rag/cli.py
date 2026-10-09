from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys

from .config import Config


def _config(args: argparse.Namespace) -> Config:
    cfg = Config()
    overrides = {}
    if args.index_dir:
        overrides["index_dir"] = args.index_dir
    if args.embed_backend:
        overrides["embed_backend"] = args.embed_backend
    if args.llm_backend:
        overrides["llm_backend"] = args.llm_backend
    if args.no_rerank:
        overrides["use_reranker"] = False
    if getattr(args, "workers", None):
        overrides["ingest_workers"] = args.workers
    return dataclasses.replace(cfg, **overrides)


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--index-dir", help="where the index lives (default ./index or $RAG_INDEX_DIR)")
    common.add_argument("--embed-backend", choices=["sentence-transformers", "hash"])
    common.add_argument("--llm-backend", choices=["anthropic", "extractive"])
    common.add_argument("--no-rerank", action="store_true", help="disable the cross-encoder reranker")

    parser = argparse.ArgumentParser(prog="rag", description="Enterprise RAG for document intelligence")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-sample", help="generate a synthetic corpus and eval set")
    p.add_argument("--out", default="data/sample/docs")
    p.add_argument("--docs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("build", parents=[common], help="ingest a directory and build the index")
    p.add_argument("data_dir")
    p.add_argument("--workers", type=int, help="parallel file-parsing processes")
    p.add_argument("--batch-size", type=int, default=512)

    p = sub.add_parser("query", parents=[common], help="ask a question")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-generate", action="store_true", help="retrieval only")

    p = sub.add_parser("eval", parents=[common], help="measure retrieval precision on a labelled set")
    p.add_argument("eval_file")
    p.add_argument("--ks", default="1,3,5,10")
    p.add_argument("--report", help="write full metrics as JSON to this path")

    p = sub.add_parser("serve", parents=[common], help="run the HTTP API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "make-sample":
        from .sample_data import generate

        path = generate(args.out, args.docs, args.seed)
        print(f"wrote {args.docs} docs to {args.out}; eval set: {path}")
        return 0

    cfg = _config(args)

    if args.cmd == "build":
        from .pipeline import RAGPipeline

        def progress(n: int) -> None:
            print(f"\r  indexed {n:,} chunks", end="", file=sys.stderr, flush=True)

        pipe = RAGPipeline.build(args.data_dir, cfg, batch_size=args.batch_size, progress=progress)
        print(f"\nbuilt index at {cfg.index_dir}: {pipe.store.count():,} chunks / {pipe.store.count_docs():,} docs")
        return 0

    if args.cmd == "query":
        from .pipeline import RAGPipeline

        pipe = RAGPipeline.load(cfg)
        result = pipe.answer(args.question, final_k=args.top_k, generate=not args.no_generate)
        if result["answer"]:
            print(result["answer"], "\n")
        for i, s in enumerate(result["sources"], 1):
            print(f"[{i}] {s['source']} (part {s['position']}) score={s['score']}")
        print(f"\ntimings: {result['timings_ms']}")
        return 0

    if args.cmd == "eval":
        from .evaluation import compare_modes, format_table, load_eval_set
        from .pipeline import RAGPipeline

        ks = tuple(int(x) for x in args.ks.split(","))
        pipe = RAGPipeline.load(cfg)
        items = load_eval_set(args.eval_file)
        results = compare_modes(pipe, items, ks)
        print(f"{len(items)} questions\n")
        print(format_table(results, ks))
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        return 0

    if args.cmd == "serve":
        import os

        import uvicorn

        # Hand the CLI overrides to the API process through the same env vars Config reads.
        os.environ.setdefault("RAG_INDEX_DIR", cfg.index_dir)
        os.environ["RAG_LLM_BACKEND"] = cfg.llm_backend
        os.environ["RAG_USE_RERANKER"] = "1" if cfg.use_reranker else "0"
        uvicorn.run("rag.api:app", host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
