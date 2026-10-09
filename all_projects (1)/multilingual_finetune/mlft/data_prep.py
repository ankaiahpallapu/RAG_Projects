"""Download / load instruction data, filter to the target languages, clean, language-balance, split.

    python -m mlft.data_prep --config config.yaml
    python -m mlft.data_prep --set data.source=jsonl --set data.jsonl_path=my.jsonl \
        --set 'data.columns={instruction: prompt, response: answer, lang: lang}'

Writes <processed_dir>/{train,val}.jsonl (+ stats.json), one {"lang","instruction","response"} per line.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from .config import Cfg, add_config_args, load_config
from .data_utils import allocate, clean_records, normalize_lang, stratified_split, write_jsonl


def iter_source(cfg: Cfg) -> Iterator[dict]:
    d, cols = cfg.data, cfg.data["columns"]
    if d["source"] == "jsonl":
        with open(d["jsonl_path"], encoding="utf-8") as f:
            rows = (json.loads(line) for line in f if line.strip())
            for row in rows:
                yield _row(row, cols)
    elif d["source"] == "hf":
        from datasets import load_dataset

        for row in load_dataset(d["hf_dataset"], split=d["hf_split"]):
            yield _row(row, cols)
    else:
        raise ValueError(f"data.source must be 'hf' or 'jsonl', got {d['source']!r}")


def _row(row: dict, cols: dict) -> dict:
    return {
        "lang": normalize_lang(str(row.get(cols["lang"], ""))),
        "instruction": row.get(cols["instruction"]) or "",
        "response": row.get(cols["response"]) or "",
    }


def prepare(cfg: Cfg) -> dict:
    d = cfg.data
    wanted = [normalize_lang(x) for x in cfg.languages]
    raw = [r for r in iter_source(cfg) if r["lang"] in wanted]
    cleaned = clean_records(raw, d["min_chars"], d["max_chars"])

    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in cleaned:
        by_lang[r["lang"]].append(r)
    missing = [lang for lang in wanted if not by_lang.get(lang)]
    if missing:
        print(f"WARNING: no usable examples for: {missing}. Check data.columns / language codes.")

    counts = {lang: len(by_lang.get(lang, [])) for lang in wanted}
    alloc = allocate(counts, d["max_total"], d["sampling_alpha"])
    rng = random.Random(d["seed"])
    sampled = {lang: rng.sample(by_lang[lang], alloc[lang]) for lang in wanted if alloc[lang]}

    train, val = stratified_split(sampled, d["val_fraction"], d["max_val_per_lang"], d["seed"])
    out = Path(d["processed_dir"])
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)

    stats = {
        lang: {"raw_clean": counts[lang], "selected": alloc[lang],
               "train": sum(r["lang"] == lang for r in train), "val": sum(r["lang"] == lang for r in val)}
        for lang in wanted
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    stats = prepare(cfg)
    print(f"{'lang':<6}{'available':>10}{'selected':>10}{'train':>8}{'val':>6}")
    for lang, s in stats.items():
        print(f"{lang:<6}{s['raw_clean']:>10}{s['selected']:>10}{s['train']:>8}{s['val']:>6}")
    print(f"\nwrote {cfg.data['processed_dir']}/train.jsonl and val.jsonl")


if __name__ == "__main__":
    main()
