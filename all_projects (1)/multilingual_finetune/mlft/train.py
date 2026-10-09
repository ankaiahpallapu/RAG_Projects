"""LoRA / QLoRA fine-tuning.

    python -m mlft.train --config config.yaml
    accelerate launch -m mlft.train --config config.yaml      # multi-GPU (DDP)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed)

from .config import add_config_args, load_config
from .data_utils import read_jsonl
from .formatting import build_example, has_supervised_tokens, pad_batch


def pick_dtype(want_bf16: bool) -> torch.dtype:
    if torch.cuda.is_available():
        if want_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def to_dataset(rows: list[dict], tok, max_len: int) -> Dataset:
    examples = [build_example(tok, r["instruction"], r["response"], max_len) for r in rows]
    return Dataset.from_list([e for e in examples if has_supervised_tokens(e)])


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        padded = pad_batch([{k: b[k] for k in ("input_ids", "labels", "attention_mask")} for b in batch], self.pad_id)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in padded.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    t, d = cfg.train, cfg.data
    set_seed(t["seed"])

    tok = load_tokenizer(cfg.base_model)
    train_rows = read_jsonl(Path(d["processed_dir"]) / "train.jsonl")
    val_rows = read_jsonl(Path(d["processed_dir"]) / "val.jsonl")
    train_ds = to_dataset(train_rows, tok, d["max_seq_len"])
    val_ds = to_dataset(val_rows, tok, d["max_seq_len"])
    print(f"train examples: {len(train_ds):,}  val examples: {len(val_ds):,}")

    dtype = pick_dtype(t["bf16"])
    quant = None
    if t["qlora"]:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=dtype, quantization_config=quant, trust_remote_code=True
    )
    model.config.use_cache = False
    if t["qlora"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=t["gradient_checkpointing"])
    elif t["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # needed so checkpointing works with frozen embeddings + LoRA

    lora = cfg.lora
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    out = Path(cfg.output_dir)
    args_tr = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=t["epochs"],
        max_steps=t["max_steps"],
        learning_rate=t["lr"],
        per_device_train_batch_size=t["per_device_batch"],
        per_device_eval_batch_size=t["per_device_batch"],
        gradient_accumulation_steps=t["grad_accum"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["scheduler"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        group_by_length=t["group_by_length"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["eval_steps"],
        save_total_limit=t["save_total_limit"],
        logging_steps=t["logging_steps"],
        report_to=t["report_to"],
        seed=t["seed"],
        remove_unused_columns=False,
        gradient_checkpointing=False,  # already enabled on the model above
        optim="paged_adamw_8bit" if t["qlora"] else "adamw_torch",
    )
    trainer = Trainer(
        model=model, args=args_tr, train_dataset=train_ds, eval_dataset=val_ds, data_collator=Collator(tok.pad_token_id)
    )
    resume = any((out / "checkpoints").glob("checkpoint-*")) if (out / "checkpoints").exists() else False
    trainer.train(resume_from_checkpoint=True if resume else None)

    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    metrics = trainer.evaluate()
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved adapter to {adapter_dir}; final eval loss {metrics.get('eval_loss'):.4f}")


if __name__ == "__main__":
    main()
