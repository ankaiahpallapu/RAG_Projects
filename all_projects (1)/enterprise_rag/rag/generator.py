from __future__ import annotations

import re
from typing import Iterator

from .bm25 import tokenize
from .config import Config
from .schema import Hit

SYSTEM_PROMPT = (
    "You are an enterprise document-intelligence assistant. Answer strictly from the numbered context "
    "passages provided. Cite the passages that support each statement with bracketed numbers such as "
    "[1] or [2][3]. If the passages do not contain the answer, say you could not find it in the "
    "provided documents. Never use outside knowledge and never invent citations."
)


def build_prompt(question: str, hits: list[Hit], max_chars: int) -> str:
    blocks: list[str] = []
    used = 0
    for i, h in enumerate(hits, 1):
        block = f"[{i}] (source: {h.chunk.source}, part {h.chunk.position})\n{h.chunk.text}"
        if blocks and used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n\nAnswer:"


class AnthropicGenerator:
    def __init__(self, model: str, max_tokens: int, max_context_chars: int):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model, self.max_tokens, self.max_context_chars = model, max_tokens, max_context_chars

    def _kwargs(self, question: str, hits: list[Hit]) -> dict:
        return dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(question, hits, self.max_context_chars)}],
        )

    def generate(self, question: str, hits: list[Hit]) -> str:
        resp = self.client.messages.create(**self._kwargs(question, hits))
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    def stream(self, question: str, hits: list[Hit]) -> Iterator[str]:
        with self.client.messages.stream(**self._kwargs(question, hits)) as s:
            yield from s.text_stream


class ExtractiveGenerator:
    """No-LLM fallback: returns the sentences most lexically similar to the question, with citations."""

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[Hit]) -> str:
        q = set(tokenize(question))
        scored: list[tuple[float, int, str]] = []
        for i, h in enumerate(hits, 1):
            for sent in re.split(r"(?<=[.!?])\s+|\n+", h.chunk.text):
                toks = set(tokenize(sent))
                if toks and q:
                    scored.append((len(q & toks) / (len(q) ** 0.5 * len(toks) ** 0.5), i, sent.strip()))
        scored.sort(key=lambda t: -t[0])
        best = [s for s in scored[: self.max_sentences] if s[0] > 0]
        if not best:
            return "I could not find an answer in the provided documents."
        return " ".join(f"{sent} [{i}]" for _, i, sent in best)

    def stream(self, question: str, hits: list[Hit]) -> Iterator[str]:
        yield self.generate(question, hits)


def make_generator(cfg: Config):
    if cfg.llm_backend == "anthropic":
        return AnthropicGenerator(cfg.llm_model, cfg.max_tokens, cfg.max_context_chars)
    if cfg.llm_backend == "extractive":
        return ExtractiveGenerator()
    raise ValueError(f"unknown llm_backend: {cfg.llm_backend!r}")
