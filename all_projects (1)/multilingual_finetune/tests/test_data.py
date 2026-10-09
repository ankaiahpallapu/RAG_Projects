"""Pure-Python tests (no GPU, no model downloads):  python tests/test_data.py"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlft.config import load_config  # noqa: E402
from mlft.data_prep import prepare  # noqa: E402
from mlft.data_utils import allocate, clean_records, normalize_lang, stratified_split, read_jsonl  # noqa: E402
from mlft.sample_data import generate  # noqa: E402
from mlft.data_utils import write_jsonl  # noqa: E402


def test_allocate():
    counts = {"a": 10, "b": 1000, "c": 1000}
    # alpha=0 -> equal shares, but 'a' can only supply 10; its unused share goes to b and c
    got = allocate(counts, 600, alpha=0.0)
    assert got["a"] == 10 and got["b"] + got["c"] == 590 and abs(got["b"] - got["c"]) <= 1, got

    # alpha=1 -> proportional to size
    got = allocate({"x": 100, "y": 300}, 200, alpha=1.0)
    assert got == {"x": 50, "y": 150}, got

    # alpha=0.5 -> small languages get MORE than their natural share
    nat = allocate({"x": 100, "y": 10_000}, 1000, alpha=1.0)
    bal = allocate({"x": 100, "y": 10_000}, 1000, alpha=0.5)
    assert bal["x"] > nat["x"], (nat, bal)

    # invariants over many random cases
    import random

    rng = random.Random(0)
    for _ in range(300):
        c = {f"l{i}": rng.randint(0, 500) for i in range(rng.randint(1, 8))}
        budget = rng.randint(0, 3000)
        a = allocate(c, budget, rng.choice([0.0, 0.3, 0.5, 1.0]))
        assert sum(a.values()) == min(budget, sum(c.values())), (c, budget, a)
        assert all(0 <= a[k] <= c[k] for k in c), (c, a)
    print("ok  allocate (caps, redistribution, alpha behaviour, 300 random invariants)")


def test_clean_and_split():
    rows = [
        {"lang": "eng", "instruction": "  Hello there  ", "response": "Hi, how can I help?"},
        {"lang": "eng", "instruction": "Hello there", "response": "Hi, how can I help?"},  # duplicate after strip
        {"lang": "eng", "instruction": "x", "response": "too short instruction"},
        {"lang": "fra", "instruction": "Bonjour tout le monde", "response": "é" * 20},
        {"lang": "fra", "instruction": "Bonjour tout le monde", "response": "é" * 20},  # NFC-equal duplicate
    ]
    out = clean_records(rows, min_chars=8, max_chars=100)
    assert len(out) == 2, out
    assert out[0]["instruction"] == "Hello there"

    by_lang = {"eng": [{"i": k} for k in range(100)], "fra": [{"i": k} for k in range(1)]}
    train, val = stratified_split(by_lang, 0.1, max_val_per_lang=5, seed=1)
    assert sum(1 for r in val) == 5 and len(train) == 96  # eng: 5 val (capped), fra: 1 row -> stays in train
    print("ok  clean_records (strip / min-length / NFC dedupe) + stratified_split")


def test_normalize_and_config():
    assert normalize_lang("en") == "eng" and normalize_lang("Hindi") == "hin" and normalize_lang("TEL") == "tel"
    cfg = load_config(None, ["train.epochs=3", "train.lr=1e-4", "languages=[eng, hin]", "data.columns.lang=language"])
    assert cfg.train.epochs == 3 and cfg.train.lr == 1e-4 and cfg.languages == ["eng", "hin"]
    assert cfg.data.columns.lang == "language" and cfg.data.columns.instruction == "inputs"
    assert isinstance(cfg.train.lr, float) and isinstance(cfg.train.epochs, int)

    # YAML 1.1 reads `2e-4` as a string; a config *file* written that way must still yield a float
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "c.yaml"
        f.write_text("train:\n  lr: 2e-4\n  eval_steps: 50\nbase_model: org/model-7b\n", encoding="utf-8")
        cfg = load_config(str(f))
        assert cfg.train.lr == 2e-4 and isinstance(cfg.train.lr, float) and cfg.train.eval_steps == 50
        assert cfg.base_model == "org/model-7b"  # non-numeric strings are untouched
    print("ok  normalize_lang + config overrides (incl. YAML '1e-4' float pitfall)")


def test_prepare_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_jsonl(tmp / "s.jsonl", generate(n_per_lang=80))
        cfg = load_config(
            None,
            [
                "data.source=jsonl",
                f"data.jsonl_path={tmp / 's.jsonl'}",
                f"data.processed_dir={tmp / 'out'}",
                "data.columns={instruction: instruction, response: response, lang: lang}",
                "data.max_total=300",
                "data.val_fraction=0.1",
            ],
        )
        stats = prepare(cfg)
        assert set(stats) == set(cfg.languages)
        train, val = read_jsonl(tmp / "out/train.jsonl"), read_jsonl(tmp / "out/val.jsonl")
        assert len(train) + len(val) == 300, (len(train), len(val))
        assert {r["lang"] for r in val} == set(cfg.languages)  # every language is represented in validation
        assert max(s["selected"] for s in stats.values()) - min(s["selected"] for s in stats.values()) <= 1
        assert json.loads((tmp / "out/stats.json").read_text())["hin"]["val"] > 0
        print("ok  prepare(): balanced selection, stratified split, files written")


if __name__ == "__main__":
    test_allocate()
    test_clean_and_split()
    test_normalize_and_config()
    test_prepare_end_to_end()
    print("\nALL PASSED")
