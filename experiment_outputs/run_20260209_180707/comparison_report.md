# Multi-Mode RAG Experiment Report

**Run ID**: 20260209_180707
**Language**: en

## Results Summary

| Metric | Hybrid | KG | KG-Ctx | Hybrid+KG | Hybrid+KG-Ctx |
|--------|--------|-------|--------|-----------|---------------|
| ROUGELScore | 0.4092 | 0.0998 | 0.1592 | 0.4047 | 0.4031 |
| Sentences_Precision | 0.1585 | 0.0000 | 0.0000 | 0.1585 | 0.1585 |
| Sentences_Recall | 0.5600 | 0.0000 | 0.0000 | 0.5600 | 0.5600 |
| Words_Precision | 0.3273 | 0.1795 | 0.1467 | 0.3273 | 0.3273 |
| Words_Recall | 0.7398 | 0.0745 | 0.3557 | 0.7398 | 0.7398 |
| completeness | 0.4232 | 0.0333 | 0.1374 | 0.4399 | 0.4436 |
| hallucination | 0.4879 | 0.4570 | 0.4453 | 0.4879 | 0.4842 |
| irrelevance | 0.0889 | 0.5097 | 0.4173 | 0.0722 | 0.0722 |

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