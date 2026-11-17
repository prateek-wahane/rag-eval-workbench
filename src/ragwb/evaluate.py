"""Evaluation harness.

Retrieval eval (free, offline): hit@k, recall@k, MRR per config and per question type.
Generation eval (calls an LLM): correctness and faithfulness (LLM-as-judge),
abstention on unanswerable questions, citation validity, latency and cost.
"""
from __future__ import annotations

import json
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass

from .config import GOLDEN_PATH, RESULTS_DIR
from .generate import answer, build_context
from .index import VectorIndex


@dataclass(frozen=True)
class GoldenItem:
    id: str
    type: str  # single | multi | confusable | unanswerable
    question: str
    expected_docs: list[str]
    reference: str


def load_golden(path=GOLDEN_PATH) -> list[GoldenItem]:
    with open(path, encoding="utf-8") as f:
        return [GoldenItem(**json.loads(line)) for line in f if line.strip()]


def _unique_docs(retrieved) -> list[str]:
    seen: list[str] = []
    for chunk, _ in retrieved:
        if chunk.doc_id not in seen:
            seen.append(chunk.doc_id)
    return seen


def _mean(xs) -> float:
    xs = list(xs)
    return round(statistics.mean(xs), 3) if xs else float("nan")


# ---------------------------------------------------------------- retrieval

def eval_retrieval(config: str, k: int, items: list[GoldenItem]) -> dict:
    index = VectorIndex.load(config)
    rows = []
    for item in items:
        if not item.expected_docs:
            continue  # unanswerable questions have no "right" document
        start = time.perf_counter()
        docs = _unique_docs(index.search(item.question, k))
        latency_ms = (time.perf_counter() - start) * 1000
        expected = set(item.expected_docs)
        ranks = [i for i, d in enumerate(docs, start=1) if d in expected]
        rows.append({
            "id": item.id,
            "type": item.type,
            "hit": 1.0 if ranks else 0.0,
            "recall": len(expected & set(docs)) / len(expected),
            "mrr": 1.0 / ranks[0] if ranks else 0.0,
            "latency_ms": latency_ms,
            "retrieved_docs": docs,
            "expected_docs": item.expected_docs,
        })
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    return {
        "config": config,
        "k": k,
        "n_chunks": len(index.chunks),
        "hit@k": _mean(r["hit"] for r in rows),
        "recall@k": _mean(r["recall"] for r in rows),
        "mrr": _mean(r["mrr"] for r in rows),
        "latency_ms": _mean(r["latency_ms"] for r in rows),
        "recall_by_type": {t: _mean(r["recall"] for r in rs) for t, rs in sorted(by_type.items())},
        "misses": [r for r in rows if r["recall"] < 1.0],
    }


def retrieval_report(results: list[dict]) -> str:
    types = sorted({t for r in results for t in r["recall_by_type"]})
    head = "| Config | Chunks | Hit@k | Recall@k | MRR | " + " | ".join(f"Recall ({t})" for t in types) + " | Query ms |"
    sep = "|" + "---|" * (6 + len(types))
    lines = [head, sep]
    for r in sorted(results, key=lambda r: (-r["recall@k"], -r["mrr"])):
        by_t = " | ".join(f"{r['recall_by_type'].get(t, float('nan')):.2f}" for t in types)
        lines.append(
            f"| `{r['config']}` | {r['n_chunks']} | {r['hit@k']:.2f} | {r['recall@k']:.2f} | {r['mrr']:.2f} | {by_t} | {r['latency_ms']:.1f} |"
        )
    k = results[0]["k"] if results else "?"
    out = [f"## Retrieval results (k={k})", "", *lines, ""]
    for r in results:
        if r["misses"]:
            out.append(f"**Misses for `{r['config']}`:** " + ", ".join(
                f"{m['id']} (wanted {'+'.join(m['expected_docs'])}, got {', '.join(m['retrieved_docs'][:3])})"
                for m in r["misses"]))
            out.append("")
    return "\n".join(out)


# --------------------------------------------------------------- generation

JUDGE_SYSTEM = """You grade answers from a retrieval-augmented assistant. Reply with ONLY a JSON object:
{"correct": true|false, "faithful": true|false, "reason": "<one short sentence>"}

- correct: the answer agrees with the reference answer on the key facts (extra correct detail is fine).
  If the reference says the assistant should not know, "correct" means the answer declined to answer.
- faithful: every claim in the answer is supported by the SOURCES shown. A pure "I don't know" is faithful."""


def _judge(judge_llm, item: GoldenItem, ans) -> dict:
    user = (
        f"SOURCES:\n{build_context(ans.retrieved)}\n\nQUESTION: {item.question}\n\n"
        f"REFERENCE ANSWER: {item.reference}\n\nASSISTANT ANSWER: {ans.text}"
    )
    res = judge_llm.complete(JUDGE_SYSTEM, user, max_tokens=4000)
    match = re.search(r"\{.*\}", res.text, re.S)
    try:
        verdict = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        verdict = {}
    return {
        "correct": bool(verdict.get("correct", False)),
        "faithful": bool(verdict.get("faithful", False)),
        "reason": verdict.get("reason", f"unparseable judge output: {res.text[:120]}"),
        "judge_cost_usd": res.cost_usd,
    }


def eval_generation(config: str, k: int, items: list[GoldenItem], llm, judge_llm, limit: int | None = None) -> dict:
    index = VectorIndex.load(config)
    rows = []
    for item in items[:limit] if limit else items:
        ans = answer(item.question, index, llm, k)
        verdict = _judge(judge_llm, item, ans)
        answerable = bool(item.expected_docs)
        rows.append({
            "id": item.id,
            "type": item.type,
            "question": item.question,
            "answer": ans.text,
            "abstained": ans.abstained,
            "answerable": answerable,
            "citations_ok": ans.abstained or (bool(ans.citations) and not ans.invalid_citations),
            "cited_right_doc": bool(set(ans.cited_doc_ids) & set(item.expected_docs)) if answerable else None,
            **verdict,
            "latency_s": round(ans.llm.latency_s, 2),
            "input_tokens": ans.llm.input_tokens,
            "output_tokens": ans.llm.output_tokens,
            "cost_usd": ans.llm.cost_usd,
            "model": ans.llm.model,
        })
        print(f"  {item.id} {'OK ' if verdict['correct'] else 'BAD'} {ans.text[:90]!r}", flush=True)

    ans_rows = [r for r in rows if r["answerable"]]
    una_rows = [r for r in rows if not r["answerable"]]
    latencies = sorted(r["latency_s"] for r in rows)
    return {
        "config": config,
        "k": k,
        "model": rows[0]["model"] if rows else "",
        "n": len(rows),
        "correctness": _mean(r["correct"] for r in ans_rows),
        "faithfulness": _mean(r["faithful"] for r in rows if not r["abstained"]),
        "abstain_when_unanswerable": _mean(r["abstained"] for r in una_rows),
        "false_abstain_rate": _mean(r["abstained"] for r in ans_rows),
        "citation_validity": _mean(r["citations_ok"] for r in rows),
        "cited_right_doc": _mean(r["cited_right_doc"] for r in ans_rows if not r["abstained"]),
        "latency_p50_s": latencies[len(latencies) // 2] if latencies else None,
        "latency_p90_s": latencies[int(len(latencies) * 0.9) - 1] if latencies else None,
        "answer_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        "judge_cost_usd": round(sum(r["judge_cost_usd"] for r in rows), 4),
        "cost_per_1k_answers_usd": round(1000 * sum(r["cost_usd"] for r in rows) / max(len(rows), 1), 2),
        "rows": rows,
    }


def generation_report(results: list[dict]) -> str:
    lines = [
        "## Generation results",
        "",
        "| Config | Model | N | Correct | Faithful | Abstains when it should | False abstain | Valid citations | Cited right doc | p50 s | p90 s | $ / 1k answers |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r['config']}` | {r['model']} | {r['n']} | {r['correctness']:.2f} | {r['faithfulness']:.2f} | "
            f"{r['abstain_when_unanswerable']:.2f} | {r['false_abstain_rate']:.2f} | {r['citation_validity']:.2f} | "
            f"{r['cited_right_doc']:.2f} | {r['latency_p50_s']} | {r['latency_p90_s']} | {r['cost_per_1k_answers_usd']} |"
        )
    lines.append("")
    for r in results:
        bad = [x for x in r["rows"] if not x["correct"] or not x["faithful"]]
        if bad:
            lines.append(f"**Failures for `{r['config']}`:**")
            lines += [f"- {x['id']} ({x['type']}): {x['reason']}" for x in bad]
            lines.append("")
    return "\n".join(lines)


def save(name: str, payload, markdown: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RESULTS_DIR / f"{name}.md").write_text(markdown + "\n", encoding="utf-8")
