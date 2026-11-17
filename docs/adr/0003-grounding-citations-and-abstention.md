# ADR 0003: Numbered-source citations and an explicit "I don't know"

**Status:** Accepted

## Context
In customer-facing policy answers, a confident wrong answer (wrong refund window, wrong warranty) costs more than no answer. Reviewers need to trace every claim to a source.

## Decision
- Retrieved chunks are numbered `[1]..[k]`. The model must end each factual sentence with its source numbers.
- If the sources don't contain the answer, the model must reply with one fixed sentence (`ABSTAIN`), which code can detect.
- Specific policies (B-Stock, Pro Installer, holiday window) are preferred over general rules. The golden set has a "confusable" slice to test this.
- Citations are checked in code (they must exist and point inside 1..k), and the cited documents are checked against the golden set.

## Why not the API's native citations feature?
Native document citations are a good production option on Claude. The prompt-based form here keeps the pipeline provider-neutral, so the same eval runs on a local Ollama model and on Claude.

## Consequences
- Tracked metrics: citation validity, cited-right-document, abstain-when-unanswerable and false-abstain rate.
- Small local models follow the citation format less reliably. That gap is measured, not assumed.
