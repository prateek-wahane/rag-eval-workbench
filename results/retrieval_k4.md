## Retrieval results (k=4)

| Config | Chunks | Hit@k | Recall@k | MRR | Recall (confusable) | Recall (multi) | Recall (single) | Query ms |
|---|---|---|---|---|---|---|---|---|
| `fixed__minilm` | 43 | 1.00 | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | 34.8 |
| `fixed__bge-small` | 43 | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 12.5 |
| `heading__bge-small` | 63 | 1.00 | 0.97 | 0.97 | 1.00 | 0.67 | 1.00 | 74.6 |
| `heading__minilm` | 63 | 1.00 | 0.97 | 0.97 | 1.00 | 0.67 | 1.00 | 43.4 |

**Misses for `heading__bge-small`:** q25 (wanted product-mesh-router+warranty, got product-mesh-router), q26 (wanted product-speaker+warranty, got product-speaker)

**Misses for `heading__minilm`:** q25 (wanted product-mesh-router+warranty, got product-mesh-router), q33 (wanted loyalty-program+shipping, got shipping, payments-financing)

