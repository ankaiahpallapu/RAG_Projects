"""Latency benchmark: baseline vs optimised serving of the SAME fine-tuned model, per language.

    python -m mlft.benchmark --config config.yaml \
        --baseline-dtype float32 --baseline-adapter outputs/run1/adapter \
        --optimized-model outputs/run1/merged --optimized-dtype bfloat16 --batch-size 8

Baseline  = base model + unmerged LoRA adapter (what training leaves you with), at --baseline-dtype.
Optimised = merged checkpoint at --optimized-dtype, batched generation, optional torch.compile.
Both use greedy decoding and a forced token count, so the work per request is identical.
The reported reduction is YOUR measured number on YOUR hardware: it depends on the baseline you choose.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from .config import add_config_args, load_config
from .data_utils import read_jsonl
from .formatting import render_prompt
from .train import load_tokenizer

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load(model_name: str, dtype: torch.dtype, adapter: str | None, compile_model: bool):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)  # deliberately NOT merged
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")
    return model


@torch.no_grad()
def time_generate(model, tok, prompts: list[str], new_tokens: int, batch_size: int, warmup: int = 2) -> list[float]:
    """Seconds per *request*, measuring each batch and dividing by batch size, plus per-batch wall time."""
    tok.padding_side = "left"  # decoder-only batching needs left padding
    device = next(model.parameters()).device
    batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    wall: list[float] = []
    for n, batch in enumerate(batches[: warmup] + batches):
        enc = tok(batch, return_tensors="pt", padding=True).to(device)
        sync()
        t0 = time.perf_counter()
        model.generate(
            **enc, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
        )
        sync()
        if n >= warmup:
            wall.append(time.perf_counter() - t0)
    return wall


def summarize(wall: list[float], batch_size: int, new_tokens: int) -> dict:
    s = sorted(wall)
    p = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {
        "p50_s": statistics.median(s), "p95_s": p(0.95),
        "tokens_per_s": batch_size * new_tokens / statistics.mean(s),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--baseline-model", help="default: base_model from config")
    ap.add_argument("--baseline-adapter", help="adapter dir; makes the baseline an UNMERGED LoRA model")
    ap.add_argument("--baseline-dtype", default="float32", choices=DTYPES)
    ap.add_argument("--baseline-batch-size", type=int, default=1)
    ap.add_argument("--optimized-model", help="default: <output_dir>/merged")
    ap.add_argument("--optimized-dtype", default="bfloat16", choices=DTYPES)
    ap.add_argument("--compile", action="store_true", help="torch.compile the optimised model (CUDA)")
    ap.add_argument("--batch-size", type=int, default=8, help="optimised batch size")
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--prompts-per-lang", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    tok = load_tokenizer(cfg.base_model)
    rows = read_jsonl(Path(cfg.data["processed_dir"]) / "val.jsonl")
    by_lang: dict[str, list[str]] = {}
    for r in rows:
        by_lang.setdefault(r["lang"], [])
        if len(by_lang[r["lang"]]) < args.prompts_per_lang:
            by_lang[r["lang"]].append(render_prompt(tok, r["instruction"]))

    configs = {
        "baseline": (args.baseline_model or cfg.base_model, args.baseline_dtype, args.baseline_adapter, False, args.baseline_batch_size),
        "optimized": (args.optimized_model or str(Path(cfg.output_dir) / "merged"), args.optimized_dtype, None, args.compile, args.batch_size),
    }
    results: dict[str, dict[str, dict]] = {}
    for name, (model_name, dtype, adapter, comp, bs) in configs.items():
        print(f"\n[{name}] {model_name} dtype={dtype} adapter={adapter} batch={bs} compile={comp}")
        model = load(model_name, DTYPES[dtype], adapter, comp)
        results[name] = {}
        for lang, prompts in by_lang.items():
            wall = time_generate(model, tok, prompts, args.new_tokens, bs)
            # per-request latency: a batch serves `bs` requests in `wall` seconds
            results[name][lang] = summarize(wall, bs, args.new_tokens) | {"per_request_s": statistics.mean(wall) / bs}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'lang':<6}{'base s/req':>12}{'opt s/req':>11}{'reduction':>11}{'base tok/s':>12}{'opt tok/s':>11}")
    reductions = []
    for lang in by_lang:
        b, o = results["baseline"][lang], results["optimized"][lang]
        red = 1 - o["per_request_s"] / b["per_request_s"]
        reductions.append(red)
        print(f"{lang:<6}{b['per_request_s']:>12.3f}{o['per_request_s']:>11.3f}{red * 100:>10.1f}%"
              f"{b['tokens_per_s']:>12.1f}{o['tokens_per_s']:>11.1f}")
    print(f"\nmean per-request latency reduction: {statistics.mean(reductions) * 100:.1f}%")


if __name__ == "__main__":
    main()
