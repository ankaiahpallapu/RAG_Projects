"""Per-language validation loss / perplexity: base model vs fine-tuned adapter.

    python -m mlft.evaluate --config config.yaml
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from .config import add_config_args, load_config
from .data_utils import read_jsonl
from .formatting import build_example, has_supervised_tokens, pad_batch
from .train import load_tokenizer, pick_dtype


@torch.no_grad()
def per_language_loss(model, tok, rows: list[dict], max_len: int, batch_size: int = 8) -> dict[str, dict]:
    """Token-weighted mean cross-entropy on response tokens only, per language."""
    device = next(model.parameters()).device
    model.eval()
    tot = defaultdict(float)
    cnt = defaultdict(int)
    by_lang = defaultdict(list)
    for r in rows:
        ex = build_example(tok, r["instruction"], r["response"], max_len)
        if has_supervised_tokens(ex):
            by_lang[r["lang"]].append(ex)
    for lang, exs in by_lang.items():
        exs.sort(key=lambda e: len(e["input_ids"]))
        for i in range(0, len(exs), batch_size):
            b = pad_batch(exs[i : i + batch_size], tok.pad_token_id)
            ids = torch.tensor(b["input_ids"], device=device)
            mask = torch.tensor(b["attention_mask"], device=device)
            labels = torch.tensor(b["labels"], device=device)
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            tgt = labels[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="sum"
            )
            tot[lang] += float(loss)
            cnt[lang] += int((tgt != -100).sum())
    return {
        lang: {"loss": tot[lang] / cnt[lang], "ppl": math.exp(min(tot[lang] / cnt[lang], 50)), "tokens": cnt[lang]}
        for lang in by_lang
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--adapter", help="adapter dir (default <output_dir>/adapter)")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)

    tok = load_tokenizer(cfg.base_model)
    rows = read_jsonl(Path(cfg.data["processed_dir"]) / "val.jsonl")
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=pick_dtype(True), trust_remote_code=True)
    if torch.cuda.is_available():
        model.to("cuda")
    max_len = cfg.data["max_seq_len"]

    base = per_language_loss(model, tok, rows, max_len, args.batch_size)
    model = PeftModel.from_pretrained(model, args.adapter or str(Path(cfg.output_dir) / "adapter"))
    tuned = per_language_loss(model, tok, rows, max_len, args.batch_size)

    print(f"{'lang':<6}{'base ppl':>10}{'tuned ppl':>11}{'change':>9}{'tokens':>9}")
    for lang in sorted(tuned):
        b, t = base[lang]["ppl"], tuned[lang]["ppl"]
        print(f"{lang:<6}{b:>10.2f}{t:>11.2f}{(t / b - 1) * 100:>8.1f}%{tuned[lang]['tokens']:>9}")


if __name__ == "__main__":
    main()
