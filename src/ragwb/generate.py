"""Citation-grounded answer generation with an explicit "I don't know" path."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .chunking import Chunk
from .index import VectorIndex
from .llm import LLMResult

ABSTAIN = "I don't know based on the available documents."

SYSTEM_PROMPT = f"""You answer customer questions for Brightleaf Home using ONLY the numbered sources provided.

Rules:
- Every factual sentence must end with the source number(s) it came from, like [1] or [2][3].
- If the sources do not contain the answer, reply with exactly: {ABSTAIN}
- Do not use outside knowledge, and do not guess numbers, prices or policies.
- When a specific policy (e.g. B-Stock, Pro Installer, holiday returns) applies, prefer it over the general rule and say so.
- Keep answers to 1-3 sentences."""


@dataclass
class Answer:
    question: str
    text: str
    retrieved: list[tuple[Chunk, float]]
    citations: list[int] = field(default_factory=list)  # 1-based source numbers the answer cites
    abstained: bool = False
    llm: LLMResult | None = None

    @property
    def invalid_citations(self) -> list[int]:
        return [c for c in self.citations if not 1 <= c <= len(self.retrieved)]

    @property
    def cited_doc_ids(self) -> list[str]:
        return sorted({self.retrieved[c - 1][0].doc_id for c in self.citations if 1 <= c <= len(self.retrieved)})


def build_context(retrieved: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{i}] {chunk.text}" for i, (chunk, _) in enumerate(retrieved, start=1))


def parse_citations(text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})


def is_abstention(text: str) -> bool:
    t = text.lower()
    return "i don't know" in t or "i do not know" in t


def answer(question: str, index: VectorIndex, llm, k: int) -> Answer:
    retrieved = index.search(question, k)
    user = f"Sources:\n{build_context(retrieved)}\n\nQuestion: {question}"
    result = llm.complete(SYSTEM_PROMPT, user)
    text = result.text if not result.refused else ABSTAIN
    return Answer(
        question=question,
        text=text,
        retrieved=retrieved,
        citations=parse_citations(text),
        abstained=is_abstention(text),
        llm=result,
    )
