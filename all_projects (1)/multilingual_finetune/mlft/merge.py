"""Fold the LoRA adapter into the base weights for serving.

    python -m mlft.merge --config config.yaml            # -> <output_dir>/merged

An unmerged adapter adds two extra small matmuls (plus kernel launches) to every adapted linear layer on every
token. After merging, the model is an ordinary dense checkpoint: no PEFT overhead, loadable by vLLM / TGI / llama.cpp.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from .config import add_config_args, load_config
from .train import load_tokenizer


def merge(base_model: str, adapter_dir: str, out_dir: str, dtype: torch.dtype = torch.bfloat16) -> str:
    # Merge in 16-bit on a non-quantised base: merging into 4-bit weights loses precision.
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    load_tokenizer(base_model).save_pretrained(out_dir)
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--adapter")
    p.add_argument("--out")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    out = merge(
        cfg.base_model,
        args.adapter or str(Path(cfg.output_dir) / "adapter"),
        args.out or str(Path(cfg.output_dir) / "merged"),
    )
    print(f"merged model saved to {out}")


if __name__ == "__main__":
    main()
