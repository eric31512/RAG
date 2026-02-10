# Multi-Mode RAG Experiment Report

**Run ID**: 20260209_163323
**Language**: en

## Results Summary

| Metric | Hybrid | KG | KG-Ctx | Hybrid+KG | Hybrid+KG-Ctx |
|--------|--------|-------|--------|-----------|---------------|
| ROUGELScore | 0.3967 | 0.0998 | 0.1359 | 0.4013 | 0.1301 |
| Sentences_Precision | 0.1585 | 0.0000 | 0.0000 | 0.1585 | 0.0000 |
| Sentences_Recall | 0.5600 | 0.0000 | 0.0000 | 0.5600 | 0.0000 |
| Words_Precision | 0.3273 | 0.1795 | 0.0000 | 0.3273 | 0.0000 |
| Words_Recall | 0.7398 | 0.0745 | 0.0000 | 0.7398 | 0.0000 |

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