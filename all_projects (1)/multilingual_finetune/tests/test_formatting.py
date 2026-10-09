"""Tests for tokenisation/masking/padding with a fake tokenizer:  python tests/test_formatting.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlft.formatting import IGNORE, build_example, has_supervised_tokens, pad_batch  # noqa: E402


class FakeTok:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "<u>" + messages[-1]["content"] + "<a>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 250 + 1 for c in text]}  # never 0, so EOS (0) is distinguishable


def test_masking():
    tok = FakeTok()
    ex = build_example(tok, "hi", "yes", max_len=100)
    n_prompt = len("<u>hi<a>")
    assert ex["labels"][:n_prompt] == [IGNORE] * n_prompt
    assert ex["labels"][n_prompt:] == ex["input_ids"][n_prompt:]
    assert ex["input_ids"][-1] == tok.eos_token_id  # response ends with EOS and it is supervised
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])
    print("ok  prompt masked, response + EOS supervised")


def test_truncation():
    tok = FakeTok()
    ex = build_example(tok, "hi", "x" * 500, max_len=20)
    assert len(ex["input_ids"]) == 20 and has_supervised_tokens(ex)
    gone = build_example(tok, "q" * 100, "ans", max_len=20)  # prompt alone exceeds max_len
    assert not has_supervised_tokens(gone)
    print("ok  truncation + empty-supervision detection")


def test_padding():
    a = {"input_ids": [1, 2, 3], "labels": [IGNORE, 2, 3], "attention_mask": [1, 1, 1]}
    b = {"input_ids": [4], "labels": [4], "attention_mask": [1]}
    out = pad_batch([a, b], pad_id=9)
    assert out["input_ids"] == [[1, 2, 3], [4, 9, 9]]
    assert out["labels"] == [[IGNORE, 2, 3], [4, IGNORE, IGNORE]]
    assert out["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
    print("ok  right-padding with ignored labels")


if __name__ == "__main__":
    test_masking()
    test_truncation()
    test_padding()
    print("\nALL PASSED")
