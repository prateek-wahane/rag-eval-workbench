"""Embeddings (fastembed, CPU) + FAISS vector index, cached on disk per config."""
from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache

import faiss
import numpy as np
from fastembed import TextEmbedding

from .chunking import Chunk, chunk_documents
from .config import CORPUS_DIR, EMBED_MODELS, INDEX_DIR, parse_config
from .ingest import load_corpus


@lru_cache(maxsize=4)
def get_embedder(key: str) -> TextEmbedding:
    return TextEmbedding(model_name=EMBED_MODELS[key])


def _normalize(vecs: np.ndarray) -> np.ndarray:
    vecs = np.asarray(vecs, dtype="float32")
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)


def embed_passages(key: str, texts: list[str]) -> np.ndarray:
    return _normalize(np.array(list(get_embedder(key).passage_embed(texts))))


def embed_query(key: str, text: str) -> np.ndarray:
    return _normalize(np.array(list(get_embedder(key).query_embed(text))))


class VectorIndex:
    """Cosine-similarity search (inner product on normalized vectors)."""

    def __init__(self, config: str, chunks: list[Chunk], index: faiss.Index):
        self.config = config
        self.embedder = parse_config(config)[1]
        self.chunks = chunks
        self.index = index

    @classmethod
    def build(cls, config: str) -> "VectorIndex":
        chunker, embedder = parse_config(config)
        chunks = chunk_documents(load_corpus(CORPUS_DIR), chunker)
        vecs = embed_passages(embedder, [c.text for c in chunks])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        out = INDEX_DIR / config
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(out / "faiss.index"))
        (out / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=1), encoding="utf-8")
        return cls(config, chunks, index)

    @classmethod
    def load(cls, config: str) -> "VectorIndex":
        path = INDEX_DIR / config
        if not (path / "faiss.index").exists():
            return cls.build(config)
        chunks = [Chunk(**c) for c in json.loads((path / "chunks.json").read_text(encoding="utf-8"))]
        return cls(config, chunks, faiss.read_index(str(path / "faiss.index")))

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        scores, ids = self.index.search(embed_query(self.embedder, query), k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
