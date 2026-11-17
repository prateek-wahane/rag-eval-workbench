# Manual review of judge verdicts: `generation_ollama`

Every judge verdict was read by hand against the reference answer. Unanswerable questions are scored in code (see `_score_unanswerable`), so only answerable questions are listed.

## Judge marked correct, but the answer is wrong or incomplete

| Config | Q | Answer (abridged) | Why it fails |
|---|---|---|---|
| heading | q16 | "The MX3 mesh system supports up to [2][3]." | The number (200) is missing. |
| heading | q25 | "...part of the Brightleaf Pro series and comes with a warranty." | Doesn't say 5 years, which was the question. |
| heading | q33 | "Plus members get free expedited shipping." | The question also asked how fast (2 business days). |
| fixed | q31 | "Yes, you can get a price match on a B-Stock item..." | **Wrong.** B-Stock is excluded from price matching. |
| fixed | q33 | "Plus members get free expedited shipping." | The question also asked how fast. |

## Judge marked wrong: verdicts confirmed

- heading: q05, q26, q31 (said "I don't know" on answerable questions)
- fixed: q05, q10, q13, q14, q29 ("I don't know" on answerable questions), q08 (garbled by a chunk boundary)

## Judge errors on unanswerable questions (now scored in code)

- fixed: q36, q39, q40 were correct "I don't know" answers that the judge marked wrong.

## Hand-checked correctness (35 answerable questions)

| Config | Judge said | Hand-checked |
|---|---|---|
| heading | 32/35 = 0.91 | **29/35 = 0.83** |
| fixed | 29/35 = 0.83 | **27/35 = 0.77** |

**Conclusion:** the 3B judge erred in both directions. It passed 5 bad answers and failed 3 good ones, which is 8 errors in 80 verdicts. The ranking (heading > fixed) survives the review, but the absolute numbers came down. A stronger judge is needed before quoting these numbers externally.
