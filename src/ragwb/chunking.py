"""Two chunking strategies to compare.

- fixed:   sliding window of N words with overlap. Ignores document structure.
- heading: one chunk per "## " section, prefixed with "Doc title > Section" so the
           chunk carries its own context. Long sections are sub-split.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .ingest import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    out, start = [], 0
    while start < len(words):
        out.append(words[start : start + size])
        if start + size >= len(words):
            break
        start += size - overlap
    return out


def chunk_fixed(doc: Document, size: int = 60, overlap: int = 15) -> list[Chunk]:
    words = doc.text.split()
    return [
        Chunk(f"{doc.doc_id}#f{i}", doc.doc_id, doc.title, "", " ".join(w))
        for i, w in enumerate(_windows(words, size, overlap))
    ]


def chunk_by_heading(doc: Document, max_words: int = 120) -> list[Chunk]:
    # Split before every "## " heading; the part before the first one is the preamble.
    parts = re.split(r"(?m)^(?=## )", doc.text)
    chunks: list[Chunk] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        if lines[0].startswith("## "):
            section, body = lines[0][3:].strip(), "\n".join(lines[1:]).strip()
        else:
            section, body = "Overview", "\n".join(l for l in lines if not l.startswith("# ")).strip()
        if not body:
            continue
        words = body.split()
        pieces = [body] if len(words) <= max_words else [" ".join(w) for w in _windows(words, max_words, 20)]
        for j, piece in enumerate(pieces):
            text = f"{doc.title} > {section}\n{piece}"
            chunks.append(Chunk(f"{doc.doc_id}#h{len(chunks)}", doc.doc_id, doc.title, section, text))
    return chunks


def chunk_documents(docs: list[Document], strategy: str) -> list[Chunk]:
    fn = {"fixed": chunk_fixed, "heading": chunk_by_heading}[strategy]
    return [c for d in docs for c in fn(d)]
