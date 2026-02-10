# Multi-Mode RAG Experiment Report

**Run ID**: 20260210_154203
**Language**: en

## Results Summary

| Metric | Hybrid | KG | KG-Ctx | Hybrid+KG | Hybrid+KG-Ctx |
|--------|--------|-------|--------|-----------|---------------|
| ROUGELScore | 0.4075 | 0.3016 | 0.2540 | 0.3515 | 0.3460 |
| Sentences_Precision | 0.1585 | 0.0106 | 0.0297 | 0.1154 | 0.1274 |
| Sentences_Recall | 0.5600 | 0.4858 | 0.1022 | 0.4754 | 0.5254 |
| Words_Precision | 0.3273 | 0.0517 | 0.1514 | 0.2986 | 0.3066 |
| Words_Recall | 0.7398 | 0.7955 | 0.4843 | 0.6887 | 0.7083 |
| completeness | 0.3954 | 0.2519 | 0.2149 | 0.3787 | 0.3871 |
| hallucination | 0.4768 | 0.4676 | 0.4990 | 0.4657 | 0.4768 |
| irrelevance | 0.1278 | 0.2806 | 0.2861 | 0.1556 | 0.1361 |

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