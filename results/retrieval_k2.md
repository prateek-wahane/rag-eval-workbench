## Retrieval results (k=2)

| Config | Chunks | Hit@k | Recall@k | MRR | Recall (confusable) | Recall (multi) | Recall (single) | Query ms |
|---|---|---|---|---|---|---|---|---|
| `fixed__minilm` | 43 | 1.00 | 0.97 | 0.93 | 1.00 | 0.67 | 1.00 | 31.9 |
| `heading__minilm` | 63 | 1.00 | 0.96 | 0.97 | 1.00 | 0.67 | 0.98 | 44.9 |
| `fixed__bge-small` | 43 | 0.97 | 0.94 | 0.91 | 0.86 | 0.67 | 1.00 | 11.3 |
| `heading__bge-small` | 63 | 0.97 | 0.93 | 0.96 | 0.86 | 0.67 | 0.98 | 74.1 |

**Misses for `fixed__bge-small`:** q25 (wanted product-mesh-router+warranty, got product-mesh-router), q29 (wanted open-box-bstock, got returns-policy), q33 (wanted loyalty-program+shipping, got loyalty-program)

**Misses for `fixed__minilm`:** q25 (wanted product-mesh-router+warranty, got product-mesh-router), q33 (wanted loyalty-program+shipping, got loyalty-program, returns-policy)

**Misses for `heading__bge-small`:** q05 (wanted returns-policy+product-smart-lock, got product-smart-lock), q25 (wanted product-mesh-router+warranty, got product-mesh-router), q26 (wanted product-speaker+warranty, got product-speaker), q31 (wanted price-match, got open-box-bstock)

**Misses for `heading__minilm`:** q05 (wanted returns-policy+product-smart-lock, got product-smart-lock), q25 (wanted product-mesh-router+warranty, got product-mesh-router), q33 (wanted loyalty-program+shipping, got shipping)

