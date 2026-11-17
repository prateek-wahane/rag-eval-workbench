## Generation results

| Config | Model | N | Correct | Faithful | Abstains when it should | False abstain | Valid citations | Cited right doc | p50 s | p90 s | $ / 1k answers |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `heading__bge-small` | qwen2.5:3b | 40 | 0.91 | 0.97 | 0.80 | 0.09 | 1.00 | 1.00 | 6.48 | 7.57 | 0.0 |
| `fixed__bge-small` | qwen2.5:3b | 40 | 0.83 | 0.97 | 1.00 | 0.14 | 0.97 | 0.90 | 8.65 | 9.81 | 0.0 |

**Failures for `heading__bge-small`:**
- q05 (single): The assistant's answer is not aligned with the reference answer, which states smart locks cannot be returned once paired.
- q26 (multi): The answer does not address the warranty duration, which is not covered by the provided sources.
- q31 (confusable): The assistant's answer contradicts the reference answer and is not supported by the sources provided.
- q36 (unanswerable): answered an unanswerable question; judge: The answer does not directly address the question about professional installation availability.

**Failures for `fixed__bge-small`:**
- q05 (single): The assistant's response is not faithful to the sources provided.
- q08 (single): The reference answer specifies the cut-off time as 2 PM Pacific time on a business day, but the assistant's answer is not faithful to the sources provided.
- q10 (single): The answer does not directly address the question and is not supported by the given sources.
- q13 (single): The assistant's answer is not faithful to the sources provided and incorrect.
- q14 (single): The answer does not directly provide the number of points needed for a $10 reward, instead stating it can be redeemed for a $10 reward.
- q29 (confusable): The assistant's answer contradicts the reference answer and does not align with the provided sources.

