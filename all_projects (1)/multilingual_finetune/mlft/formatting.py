"""Tokenisation + batching helpers. Duck-typed on the tokenizer so they're testable without transformers."""

from __future__ import annotations

IGNORE = -100


def render_prompt(tok, instruction: str, system: str | None = None) -> str:
    """Chat-template text for the user turn, ending where the assistant's reply begins."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": instruction}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_example(tok, instruction: str, response: str, max_len: int, system: str | None = None) -> dict:
    """input_ids = prompt + response + EOS; labels mask the prompt so loss is only on the response."""
    prompt_ids = tok(render_prompt(tok, instruction, system), add_special_tokens=False)["input_ids"]
    resp_ids = tok(response, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (prompt_ids + resp_ids)[:max_len]
    labels = ([IGNORE] * len(prompt_ids) + resp_ids)[:max_len]
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def has_supervised_tokens(example: dict) -> bool:
    """False when truncation ate the whole response (such rows only waste compute)."""
    return any(x != IGNORE for x in example["labels"])


def pad_batch(batch: list[dict], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad to the longest sequence in the batch."""
    n = max(len(b["input_ids"]) for b in batch)
    out: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        pad = n - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["labels"].append(b["labels"] + [IGNORE] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
    return out
