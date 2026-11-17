"""Load markdown documents from the corpus folder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str  # file stem, e.g. "returns-policy"
    title: str  # first "# " heading
    text: str


def load_corpus(corpus_dir: Path) -> list[Document]:
    docs = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        title = next(
            (line[2:].strip() for line in text.splitlines() if line.startswith("# ")),
            path.stem,
        )
        docs.append(Document(doc_id=path.stem, title=title, text=text))
    if not docs:
        raise FileNotFoundError(f"No .md files found in {corpus_dir}")
    return docs
