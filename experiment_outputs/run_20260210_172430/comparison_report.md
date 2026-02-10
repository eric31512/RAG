# Multi-Mode RAG Experiment Report

**Run ID**: 20260210_172430
**Language**: en

## Results Summary

| Metric | Hybrid | KG | KG-Ctx | Hybrid+KG | Hybrid+KG-Ctx |
|--------|--------|-------|--------|-----------|---------------|
| ROUGELScore | 0.4064 | 0.3141 | 0.3260 | 0.4491 | 0.3200 |
| Sentences_Precision | 0.1412 | 0.0121 | 0.0171 | 0.0797 | 0.0564 |
| Sentences_Recall | 0.5735 | 0.4858 | 0.1652 | 0.6265 | 0.4871 |
| Words_Precision | 0.3160 | 0.0560 | 0.1049 | 0.1824 | 0.1697 |
| Words_Recall | 0.7325 | 0.8029 | 0.6208 | 0.7882 | 0.7387 |
| completeness | 0.4426 | 0.2574 | 0.3186 | 0.4426 | 0.3482 |
| hallucination | 0.4824 | 0.4842 | 0.4203 | 0.4824 | 0.4713 |
| irrelevance | 0.0750 | 0.2583 | 0.2611 | 0.0750 | 0.1806 |

## Mode Descriptions

| Mode | Description |
|------|-------------|
| **Hybrid** | BM25 + Vector (Dense) + Reranker |
| **KG** | Regular KG retrieval only |
| **KG-Ctx** | Contextual KG retrieval only |
| **Hybrid+KG** | Hybrid retrieval + Regular KG context |
| **Hybrid+KG-Ctx** | Hybrid retrieval + Contextual KG context |

## Files

- `output_hybrid.jsonl`: RAG predictions
- `eval_hybrid.jsonl`: Evaluation results
- `output_kg.jsonl`: RAG predictions
- `eval_kg.jsonl`: Evaluation results
- `output_kg-contextual.jsonl`: RAG predictions
- `eval_kg-contextual.jsonl`: Evaluation results
- `output_hybrid-kg.jsonl`: RAG predictions
- `eval_hybrid-kg.jsonl`: Evaluation results
- `output_hybrid-kg-contextual.jsonl`: RAG predictions
- `eval_hybrid-kg-contextual.jsonl`: Evaluation results