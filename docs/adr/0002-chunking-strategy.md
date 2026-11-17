# ADR 0002: Compare fixed-window and heading-aware chunking, decide with data

**Status:** Accepted (default: `heading`, revisit with hybrid retrieval)

## Context
Chunking is the biggest lever in a RAG system and is usually chosen by habit. Policy documents have clear `##` sections, but some answers need two documents. For example, "MX3 warranty" needs the product page ("Pro series") plus the warranty policy ("Pro = 5 years").

## Options
1. **Fixed window**: 60 words, 15-word overlap. Structure-blind, fewer and larger chunks.
2. **Heading-aware**: one chunk per section, prefixed `Doc title > Section`, so each chunk carries its own context.

## Decision
Measure both against the golden set (`results/retrieval_k*.md`) and keep both in the code.
- Heading-aware has the best **MRR** (0.96-0.97): the right document ranks first more often, which reduces noise in the prompt.
- Fixed-window has better **recall on multi-document questions** at k=4, because its larger windows bring in neighbouring facts.
- Neither solves multi-hop at k=2 (0.67 recall on multi-document questions).

## Consequences / next steps
- Default to `heading` for precision, with k=4.
- Fix multi-hop by retrieval design, not chunk size: add hybrid BM25+vector retrieval and/or query decomposition, then re-run the same eval to prove the gain.
