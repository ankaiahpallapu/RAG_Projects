from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "output_dir": "outputs/run1",
    "languages": ["eng", "hin", "spa", "fra", "deu", "tel"],
    "data": {
        "source": "hf",
        "hf_dataset": "CohereLabs/aya_dataset",
        "hf_split": "train",
        "jsonl_path": None,
        "columns": {"instruction": "inputs", "response": "targets", "lang": "language_code"},
        "processed_dir": "data/processed",
        "max_total": 60000,
        "sampling_alpha": 0.5,
        "val_fraction": 0.02,
        "max_val_per_lang": 300,
        "min_chars": 8,
        "max_chars": 6000,
        "max_seq_len": 1024,
        "seed": 42,
    },
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
    "train": {
        "epochs": 2,
        "lr": 2e-4,
        "per_device_batch": 8,
        "grad_accum": 4,
        "warmup_ratio": 0.03,
        "scheduler": "cosine",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "bf16": True,
        "gradient_checkpointing": True,
        "qlora": False,
        "group_by_length": True,
        "eval_steps": 200,
        "logging_steps": 20,
        "save_total_limit": 2,
        "max_steps": -1,
        "report_to": "none",
        "seed": 42,
    },
}


class Cfg(dict):
    """dict with attribute access: cfg.train.lr  (read-only convenience)."""

    def __getattr__(self, key: str):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Cfg(value) if isinstance(value, dict) else value


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _coerce_numbers(obj: Any) -> Any:
    """PyYAML follows YAML 1.1, where `1e-4` (no decimal point) is a *string*, so `lr: 1e-4` would silently
    reach the trainer as text. Convert any numeric-looking string back to int/float."""
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str) and _NUMBER.match(obj.strip()):
        text = obj.strip()
        return int(text) if re.fullmatch(r"[+-]?\d+", text) else float(text)
    return obj


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Cfg:
    """Defaults < YAML file < `--set a.b.c=value` overrides (values parsed as YAML: 1, 2e-4, true, [a,b])."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        _deep_update(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return Cfg(_coerce_numbers(cfg))


def add_config_args(parser) -> None:
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set train.epochs=1 (repeatable)")
