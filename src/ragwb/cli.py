"""Command line entry point: `ragwb <command>` (or `python -m ragwb.cli <command>`)."""
from __future__ import annotations

import argparse

from . import config
from .evaluate import (
    eval_generation,
    eval_retrieval,
    generation_report,
    load_golden,
    retrieval_report,
    save,
)
from .generate import answer
from .index import VectorIndex
from .llm import get_llm


def cmd_index(args):
    for name in args.configs or config.ALL_CONFIGS:
        idx = VectorIndex.build(name)
        print(f"built {name}: {len(idx.chunks)} chunks")


def cmd_ask(args):
    idx = VectorIndex.load(args.config)
    ans = answer(args.question, idx, get_llm(args.provider), args.k)
    print(ans.text, "\n")
    for i, (chunk, score) in enumerate(ans.retrieved, start=1):
        mark = "*" if i in ans.citations else " "
        print(f" {mark}[{i}] {score:.3f}  {chunk.doc_id} > {chunk.section or chunk.chunk_id}")
    r = ans.llm
    print(f"\n{r.model} | {r.input_tokens} in / {r.output_tokens} out tokens | {r.latency_s:.2f}s | ${r.cost_usd:.5f}")


def cmd_eval_retrieval(args):
    items = load_golden()
    results = [eval_retrieval(name, args.k, items) for name in args.configs or config.ALL_CONFIGS]
    md = retrieval_report(results)
    save("retrieval", results, md)
    print(md)
    if args.min_recall is not None:
        failing = [r["config"] for r in results if r["recall@k"] < args.min_recall]
        if failing:
            raise SystemExit(f"recall@{args.k} below {args.min_recall} for: {', '.join(failing)}")


def cmd_eval_generation(args):
    items = load_golden()
    llm = get_llm(args.provider)
    judge = get_llm(args.judge_provider or args.provider)
    results = []
    for name in args.configs or [config.DEFAULT_CONFIG]:
        print(f"evaluating {name} ...")
        results.append(eval_generation(name, args.k, items, llm, judge, args.limit))
    md = generation_report(results)
    save(f"generation_{args.provider}", results, md)
    print(md)


def main():
    p = argparse.ArgumentParser(prog="ragwb", description="RAG + Evaluation Workbench")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("index", help="build vector indexes")
    s.add_argument("configs", nargs="*", help=f"default: all ({', '.join(config.ALL_CONFIGS)})")
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("ask", help="ask one question")
    s.add_argument("question")
    s.add_argument("--config", default=config.DEFAULT_CONFIG)
    s.add_argument("--provider", default=config.LLM_PROVIDER, choices=["claude", "ollama"])
    s.add_argument("-k", type=int, default=config.TOP_K)
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("eval-retrieval", help="score retrieval for every config (no LLM, free)")
    s.add_argument("configs", nargs="*")
    s.add_argument("-k", type=int, default=config.TOP_K)
    s.add_argument("--min-recall", type=float, help="exit non-zero if any config scores below this (CI gate)")
    s.set_defaults(fn=cmd_eval_retrieval)

    s = sub.add_parser("eval-generation", help="score answers end to end (calls the LLM)")
    s.add_argument("configs", nargs="*")
    s.add_argument("--provider", default=config.LLM_PROVIDER, choices=["claude", "ollama"])
    s.add_argument("--judge-provider", choices=["claude", "ollama"], help="default: same as --provider")
    s.add_argument("--limit", type=int, help="only the first N questions (cheap smoke test)")
    s.add_argument("-k", type=int, default=config.TOP_K)
    s.set_defaults(fn=cmd_eval_generation)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
