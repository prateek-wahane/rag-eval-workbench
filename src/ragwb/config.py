"""Paths and the experiment matrix. Everything tunable lives here or in .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CORPUS_DIR = ROOT / "data" / "corpus"
GOLDEN_PATH = ROOT / "data" / "golden" / "questions.jsonl"
INDEX_DIR = ROOT / ".index"
RESULTS_DIR = ROOT / "results"

# Embedding models: both run on CPU via ONNX (fastembed), no GPU needed.
EMBED_MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}
CHUNKERS = ("fixed", "heading")

# A retrieval config is "<chunker>__<embedder>", e.g. "heading__bge-small".
ALL_CONFIGS = [f"{c}__{e}" for c in CHUNKERS for e in EMBED_MODELS]
DEFAULT_CONFIG = os.getenv("RAG_CONFIG", "heading__bge-small")
TOP_K = int(os.getenv("TOP_K", "4"))

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude")  # "claude" | "ollama"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "low")  # low | medium | high | xhigh | max
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


def parse_config(name: str) -> tuple[str, str]:
    chunker, embedder = name.split("__")
    if chunker not in CHUNKERS or embedder not in EMBED_MODELS:
        raise ValueError(f"Unknown config {name!r}. Choose from: {', '.join(ALL_CONFIGS)}")
    return chunker, embedder
