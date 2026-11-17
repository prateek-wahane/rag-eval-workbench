"""LLM providers behind one tiny interface, with token, latency and cost tracking.

- ClaudeLLM: Anthropic API (needs ANTHROPIC_API_KEY or `ant auth login`).
- OllamaLLM: a local model (e.g. qwen2.5:3b) - free, offline, slower on CPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from . import config

# USD per 1M tokens (input, output). Check current prices before quoting them.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    refused: bool = False


class ClaudeLLM:
    def __init__(self, model: str = config.CLAUDE_MODEL, effort: str = config.CLAUDE_EFFORT):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def complete(self, system: str, user: str, max_tokens: int = 16000) -> LLMResult:
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
        )
        start = time.perf_counter()
        if self.model in FALLBACK_MODELS:
            resp = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            resp = self.client.messages.create(**kwargs)
        latency = time.perf_counter() - start

        refused = resp.stop_reason == "refusal"
        text = "" if refused else "".join(b.text for b in resp.content if b.type == "text")
        p_in, p_out = PRICES.get(resp.model, PRICES.get(self.model, (0.0, 0.0)))
        u = resp.usage
        cost = (u.input_tokens * p_in + u.output_tokens * p_out) / 1_000_000
        return LLMResult(text.strip(), resp.model, u.input_tokens, u.output_tokens, latency, cost, refused)


class OllamaLLM:
    def __init__(self, model: str = config.OLLAMA_MODEL, host: str = config.OLLAMA_HOST):
        self.model = model
        self.host = host.rstrip("/")

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> LLMResult:
        start = time.perf_counter()
        r = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": 0, "num_predict": max_tokens},
            },
            timeout=600,
        )
        r.raise_for_status()
        data = r.json()
        return LLMResult(
            text=data["message"]["content"].strip(),
            model=self.model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_s=time.perf_counter() - start,
            cost_usd=0.0,
        )


def get_llm(provider: str | None = None):
    provider = provider or config.LLM_PROVIDER
    if provider == "claude":
        return ClaudeLLM()
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"Unknown LLM provider {provider!r} (use 'claude' or 'ollama')")
