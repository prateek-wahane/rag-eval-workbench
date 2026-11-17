# RAG + Evaluation Workbench

A small, measurable retrieval-augmented generation (RAG) system for a customer-support knowledge base. It compares **chunking strategies** and **embedding models**, generates **citation-grounded answers** with an explicit "I don't know" path, and scores everything against a **golden question set**: quality, latency and cost per answer.

Runs on a 16 GB laptop with **no GPU**. The LLM is either **Claude** (Anthropic API) or a **local model via Ollama** (free, offline).

> The knowledge base is a fictional retailer ("Brightleaf Home"): 14 policy and product documents written for this project, so it contains no real or employer data.

**How this differs from [ShopAssist](https://github.com/prateek-wahane/shopassist-agentic-rag):** ShopAssist covers *operating* a RAG system: versioned golden datasets, run history, before/after comparisons. This workbench covers the *design decisions* behind one. Which chunking and embedding pair should you pick? Can every claim be traced to a source? Does it decline when it doesn't know? What does a hosted model buy you over a free local one in quality, latency and cost?

## Why this exists

Most RAG demos show one happy-path answer. Production teams need to know:
- **Which design choice actually helps?** Chunking and embeddings are compared on the same questions.
- **Does it make things up?** Faithfulness is judged against the retrieved sources, and unanswerable questions check that it says "I don't know".
- **Can you trace every claim?** Every sentence cites a numbered source, and citations are validated in code.
- **What does it cost and how fast is it?** Tokens, p50/p90 latency and $ per 1,000 answers are reported.

## Architecture

```
data/corpus/*.md ──► ingest ──► chunk (fixed | heading) ──► embed (bge-small | MiniLM, CPU/ONNX)
                                                                   │
                                                                   ▼
question ──► embed query ──► FAISS top-k ──► numbered sources [1..k] ──► LLM (Claude | Ollama)
                                                                   │
                                     answer with [n] citations or "I don't know"
                                                                   │
data/golden/questions.jsonl ──► evaluate: retrieval metrics + LLM-as-judge + cost/latency ──► results/*.md
```

| Module | Responsibility |
|---|---|
| `src/ragwb/chunking.py` | Fixed-window vs heading-aware chunking |
| `src/ragwb/index.py` | fastembed embeddings + FAISS index, cached per config |
| `src/ragwb/generate.py` | Grounded prompt, citation parsing, abstention |
| `src/ragwb/llm.py` | Claude / Ollama providers with token, latency and cost tracking |
| `src/ragwb/evaluate.py` | Retrieval metrics, LLM-as-judge, reports |

Design decisions are recorded in [`docs/adr/`](docs/adr).

## Golden set

40 questions in [`data/golden/questions.jsonl`](data/golden/questions.jsonl):

| Type | Count | What it tests |
|---|---|---|
| single | 25 | Answer lives in one document |
| multi | 3 | Needs two documents, e.g. "MX3 warranty" = product page (Pro series) + warranty policy (Pro = 5 years) |
| confusable | 7 | A specific rule overrides a general one: B-Stock 15-day returns vs standard 30 |
| unanswerable | 5 | Not in the documents; the correct behaviour is "I don't know" |

## Results

### Retrieval (no LLM, free to run)

At **k=4**: every config finds a relevant document for every question (hit@4 = 1.00). Heading-aware chunking ranks the right document first more often (MRR 0.97 vs 0.92-0.93), but it **misses the second document on multi-document questions** (recall 0.67).

| Config | Chunks | Hit@4 | Recall@4 | MRR | Recall (multi) | Query ms |
|---|---|---|---|---|---|---|
| `fixed__minilm` | 43 | 1.00 | 1.00 | 0.93 | 1.00 | 35 |
| `fixed__bge-small` | 43 | 1.00 | 1.00 | 0.92 | 1.00 | 13 |
| `heading__bge-small` | 63 | 1.00 | 0.97 | 0.97 | 0.67 | 75 |
| `heading__minilm` | 63 | 1.00 | 0.97 | 0.97 | 0.67 | 43 |

At **k=2** the gaps widen. bge-small drops to 0.86 recall on *confusable* questions, and every config falls to 0.67 on multi-document questions. Full tables and per-question misses: [`results/retrieval_k4.md`](results/retrieval_k4.md), [`results/retrieval_k2.md`](results/retrieval_k2.md).

**Takeaway:** chunking alone can't solve multi-hop questions. The next experiment is hybrid BM25 + vector retrieval and query decomposition, measured with this same eval ([ADR 0002](docs/adr/0002-chunking-strategy.md)).

### Generation

All 40 questions, answered end to end by a **free local model** (`qwen2.5:3b` via Ollama, CPU only), k=4. Answerable questions were graded by an LLM judge (the same 3B model), and then **every verdict was checked by hand**. Unanswerable questions are scored in code, because the right behaviour there is known exactly.

| Chunking | Correct, hand-checked (35) | Correct, judge only | Says "I don't know" when it should (5) | Wrongly says "I don't know" | Valid citations | p50 latency | Input tokens / answer |
|---|---|---|---|---|---|---|---|
| **heading** | **0.83** | 0.91 | 0.80 | **3** | **1.00** | **6.5 s** | **350** |
| fixed | 0.77 | 0.83 | 1.00 | 5 | 0.97 | 8.7 s | 476 |

**Retrieval recall didn't predict answer quality.** Fixed-window chunks had *higher* recall@4 (1.00 vs 0.97) but produced *worse* answers:
- **One confident wrong answer** (q31): asked "Can I price-match a B-Stock item?", it said *yes*. The policy excludes B-Stock. This is the costliest kind of failure.
- **One garbled answer** (q08): a chunk boundary cut the sentence about the 2 PM cut-off in half.
- **More needless "I don't know" answers** (5 vs 3). The larger, noisier context made the small model give up more often. It also cost 36% more input tokens and 2 s more latency.

Heading chunks failed more safely, but not cleanly: three cautious "I don't know" answers, two answers that dropped the key number (q16, q25), and one off-topic answer to an unanswerable question (q36).

**The judge was the weakest link.** The 3B judge erred in both directions: it passed 5 wrong or incomplete answers, including the wrong "yes" on q31, and failed 3 correct "I don't know" answers. That's 8 errors in 80 verdicts, which is why the hand-checked column is the one to trust. Full review: [`results/manual_review_ollama.md`](results/manual_review_ollama.md). Raw output: [`results/generation_ollama.md`](results/generation_ollama.md).

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env          # macOS/Linux: cp .env.example .env

pytest -q                        # unit tests
ragwb index                      # build all 4 indexes (downloads ~200 MB of embedding models once)
ragwb eval-retrieval             # free, offline retrieval scores -> results/retrieval.md
ragwb ask "Can I return a B-Stock router I opened?"
ragwb eval-generation --provider ollama              # local model, free
ragwb eval-generation --provider claude --limit 5    # Claude, small paid smoke test first
```

For a free local LLM, install [Ollama](https://ollama.com) and run `ollama pull qwen2.5:3b`. For Claude, set `ANTHROPIC_API_KEY`. The model and effort are configurable in `.env`.

> **Windows note:** keep the project in a short path (e.g. `D:\Projects\rag-eval-workbench`). Some dependency paths exceed Windows' 260-character limit inside deep folders.

## Limitations (stated honestly)

- The corpus is small (14 docs, 43-63 chunks), so retrieval scores are near their ceiling. The method matters more than the absolute numbers.
- The answer judge is the same 3B local model, and it got 8 of 80 verdicts wrong. The reported numbers are hand-checked, but a stronger judge is needed before this eval can run unattended.
- No hybrid search, reranking or query rewriting yet. These are the planned next experiments.

## Roadmap

- [ ] Hybrid retrieval (BM25 + vector) and a cross-encoder reranker, compared with the current eval
- [ ] Claude vs local model on the same golden set (quality vs $ vs latency), with a stronger judge
- [x] Local-model end-to-end eval, hand-checked (heading 0.83 vs fixed 0.77 correct)
- [ ] Prompt-injection test documents ("ignore previous instructions...") added to the eval
- [ ] Streamlit demo UI and a 2-minute demo video
- [ ] GitHub Actions: run `pytest` + `eval-retrieval` on every push and fail on regression
