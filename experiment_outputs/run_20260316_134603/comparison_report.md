# Multi-Mode RAG Experiment Report

**Run ID**: 20260316_134603

**Language**: en

## Results Summary

| Metric | Regular KG | Contextual KG |
|--------|-------|-------|
| ROUGELScore | 0.4405 | 0.3937 |
| Sentences_Precision | 0.0326 | 0.0351 |
| Sentences_Recall | 0.3902 | 0.4184 |
| Words_Precision | 0.1250 | 0.1333 |
| Words_Recall | 0.7253 | 0.7510 |
| completeness | 0.2908 | 0.3769 |
| hallucination | 0.4731 | 0.5176 |
| irrelevance | 0.2361 | 0.1056 |

## Mode Descriptions

| Mode | Description |
|------|-------------|
| **Regular KG** | KG retrieval based on `prefix_original_kg_all` |
| **Contextual KG** | KG retrieval based on `original_contextual_kg_all` |

## Files

- `output_prefix_original_kg_all.jsonl`: RAG predictions
- `eval_prefix_original_kg_all.jsonl`: Evaluation results
- `output_original_contextual_kg_all.jsonl`: RAG predictions
- `eval_original_contextual_kg_all.jsonl`: Evaluation results