# ADR 0001: CPU-only embeddings (fastembed/ONNX) and FAISS

**Status:** Accepted

## Context
The workbench must run on a 16 GB laptop with no GPU, install in minutes, and be reproducible by a reviewer. The corpus is small (tens to low thousands of chunks).

## Decision
- Embeddings: `fastembed` (ONNX runtime) with `BAAI/bge-small-en-v1.5` and `all-MiniLM-L6-v2` (384 dimensions each). No PyTorch install.
- Vector store: FAISS `IndexFlatIP` on L2-normalized vectors (exact cosine search), persisted per config under `.index/`.

## Consequences
- Install is light, and query embedding plus search takes 10-75 ms on CPU.
- Exact search is fine below ~1M vectors. Beyond that, switch to an approximate index (HNSW/IVF) or a managed store.
- No metadata filtering, multi-tenancy or online updates. For production these move to Qdrant, pgvector or OpenSearch (see ADR 0004).
