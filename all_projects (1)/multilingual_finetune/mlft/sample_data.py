"""Tiny synthetic multilingual Q&A file for a pipeline dry-run (NOT for judging model quality).

    python -m mlft.sample_data --out data/sample.jsonl
    python -m mlft.data_prep --set data.source=jsonl --set data.jsonl_path=data/sample.jsonl \
        --set 'data.columns={instruction: instruction, response: response, lang: lang}'
"""

from __future__ import annotations

import argparse
import itertools
import random

from .data_utils import write_jsonl

CAPITALS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Spain", "Madrid"), ("Italy", "Rome"), ("Japan", "Tokyo"),
    ("India", "New Delhi"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Canada", "Ottawa"),
    ("Brazil", "Brasilia"), ("Norway", "Oslo"), ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Poland", "Warsaw"),
]
TEMPLATES = {
    "eng": ("What is the capital of {c}?", "The capital of {c} is {k}."),
    "spa": ("¿Cuál es la capital de {c}?", "La capital de {c} es {k}."),
    "fra": ("Quelle est la capitale de {c} ?", "La capitale de {c} est {k}."),
    "deu": ("Was ist die Hauptstadt von {c}?", "Die Hauptstadt von {c} ist {k}."),
    "hin": ("{c} की राजधानी क्या है?", "{c} की राजधानी {k} है।"),
    "tel": ("{c} రాజధాని ఏది?", "{c} రాజధాని {k}."),
}


def generate(n_per_lang: int = 60, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for lang, (q, a) in TEMPLATES.items():
        pairs = list(itertools.islice(itertools.cycle(CAPITALS), n_per_lang))
        for i, (country, capital) in enumerate(pairs):
            # a numeric tag keeps (instruction, response) pairs unique so de-duplication keeps them
            tag = f" ({rng.randint(1, 9999)})" if i >= len(CAPITALS) else ""
            rows.append({"lang": lang, "instruction": q.format(c=country) + tag, "response": a.format(c=country, k=capital)})
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sample.jsonl")
    p.add_argument("--per-lang", type=int, default=60)
    a = p.parse_args()
    print(f"wrote {write_jsonl(a.out, generate(a.per_lang))} rows to {a.out}")
