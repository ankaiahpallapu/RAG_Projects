"""Pure-Python data utilities (no torch/transformers) so they are fast to import and easy to test."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from pathlib import Path
from typing import Iterable

LANG_NAMES = {
    "eng": "English", "hin": "Hindi", "spa": "Spanish", "fra": "French", "deu": "German", "tel": "Telugu",
    "tam": "Tamil", "ben": "Bengali", "mar": "Marathi", "kan": "Kannada", "mal": "Malayalam", "urd": "Urdu",
    "arb": "Arabic", "zho": "Chinese", "jpn": "Japanese", "kor": "Korean", "por": "Portuguese",
    "rus": "Russian", "ita": "Italian", "tur": "Turkish", "vie": "Vietnamese", "ind": "Indonesian",
}
_ISO1_TO_3 = {
    "en": "eng", "hi": "hin", "es": "spa", "fr": "fra", "de": "deu", "te": "tel", "ta": "tam", "bn": "ben",
    "mr": "mar", "kn": "kan", "ml": "mal", "ur": "urd", "ar": "arb", "zh": "zho", "ja": "jpn", "ko": "kor",
    "pt": "por", "ru": "rus", "it": "ita", "tr": "tur", "vi": "vie", "id": "ind",
}
_NAME_TO_3 = {name.lower(): code for code, name in LANG_NAMES.items()}


def normalize_lang(value: str) -> str:
    """Map 'en' / 'English' / 'ENG' to the ISO 639-3 code; unknown values pass through lower-cased."""
    v = (value or "").strip().lower()
    return _ISO1_TO_3.get(v) or _NAME_TO_3.get(v) or v


def clean_records(records: Iterable[dict], min_chars: int, max_chars: int) -> list[dict]:
    """NFC-normalise, strip, length-filter and exact-deduplicate (lang, instruction, response)."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        ins = unicodedata.normalize("NFC", (r.get("instruction") or "").strip())
        res = unicodedata.normalize("NFC", (r.get("response") or "").strip())
        if not (min_chars <= len(ins) <= max_chars and min_chars <= len(res) <= max_chars):
            continue
        key = hashlib.md5(f"{r['lang']}\x00{ins}\x00{res}".encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append({"lang": r["lang"], "instruction": ins, "response": res})
    return out


def allocate(counts: dict[str, int], budget: int, alpha: float) -> dict[str, int]:
    """Split `budget` examples across languages with weights proportional to count**alpha.

    alpha=1 keeps natural proportions, alpha=0 gives every language the same share, 0.3-0.5 is the usual
    compromise that stops high-resource languages from drowning out low-resource ones. A language is
    never asked for more examples than it has; its unused share is redistributed to the others.
    """
    alloc = {lang: 0 for lang in counts}
    active = {lang for lang, c in counts.items() if c > 0}
    remaining = min(budget, sum(counts.values()))
    while active and remaining > 0:
        weights = {lang: counts[lang] ** alpha for lang in active}
        total = sum(weights.values())
        proposed = {lang: remaining * weights[lang] / total for lang in active}
        capped = [lang for lang in active if proposed[lang] >= counts[lang] - alloc[lang]]
        if capped:  # these languages can't absorb their share: take everything they have, redistribute the rest
            for lang in capped:
                give = counts[lang] - alloc[lang]
                alloc[lang] += give
                remaining -= give
                active.discard(lang)
            continue
        floors = {lang: int(proposed[lang]) for lang in active}
        leftover = remaining - sum(floors.values())
        for lang in sorted(active, key=lambda x: proposed[x] - floors[x], reverse=True)[:leftover]:
            floors[lang] += 1
        for lang, n in floors.items():
            alloc[lang] += n
        remaining = 0
    return alloc


def stratified_split(
    by_lang: dict[str, list[dict]], val_fraction: float, max_val_per_lang: int, seed: int
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    for lang in sorted(by_lang):
        rows = list(by_lang[lang])
        rng.shuffle(rows)
        n_val = 0 if len(rows) < 2 else min(max_val_per_lang, max(1, round(len(rows) * val_fraction)))
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    rng.shuffle(train)
    return train, val


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
