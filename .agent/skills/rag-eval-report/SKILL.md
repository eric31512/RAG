---
name: rag-eval-report
description: Aggregate RAG evaluation results from JSONL files and generate a comprehensive Markdown comparison report. Use this skill whenever the user wants to analyze, summarize, or compare RAG evaluation metrics (ROUGE, precision, recall, completeness, hallucination, irrelevance) from experiment output files. Also trigger when the user mentions evaluation reports, experiment results analysis, or wants to compute average metrics from eval JSONL files.
---

# RAG Evaluation Report Generator

Generate a structured Markdown report from RAG evaluation JSONL files. The report includes overall averages, standard deviations, deltas between configurations, and breakdowns by query type and domain.

## When to Use

- User wants to compute average evaluation metrics from `.jsonl` eval files
- User wants to compare two or more KG retrieval configurations
- User asks for a summary or analysis of RAG experiment results
- User mentions terms like "evaluation report", "aggregate metrics", "compare experiments"

## How It Works

1. Locate the experiment directory containing `eval_*.jsonl` files
2. Run the bundled `scripts/compute_eval_metrics.py` script
3. The script generates a Markdown report in a `result/` subdirectory

## Usage

### Step 1: Identify the experiment directory

The experiment directory should contain one or more `eval_*.jsonl` files. Each line in these files is a JSON object with the following metric fields:

- `ROUGELScore` — ROUGE-L F1 score
- `Sentences_Precision` / `Sentences_Recall` — sentence-level precision and recall
- `Words_Precision` / `Words_Recall` — word-level precision and recall
- `completeness` — fraction of key points covered
- `hallucination` — fraction of hallucinated content
- `irrelevance` — fraction of irrelevant content

Each entry also has `domain` and `query.query_type` fields used for breakdowns.

### Step 2: Run the script

```bash
python <skill-path>/scripts/compute_eval_metrics.py <experiment_directory> [--output_dir <output_dir>]
```

**Arguments:**
- `<experiment_directory>` — Path to the directory containing `eval_*.jsonl` files (required)
- `--output_dir` — Custom output directory for the report (optional, defaults to `<experiment_directory>/result/`)

**Example:**
```bash
python .agent/skills/rag-eval-report/scripts/compute_eval_metrics.py \
  experiment_outputs/run_20260316_153632
```

### Step 3: Review the output

The script produces `evaluation_report.md` in the output directory with:

1. **Overall Average Metrics** — Mean values for each metric across all configurations
2. **Standard Deviation** — Variability of each metric
3. **Delta Comparison** — When exactly 2 configurations are found, shows the difference and which is better (for hallucination and irrelevance, lower is better)
4. **Breakdown by Query Type** — Per-query-type metrics (e.g., Factual Question, Multi-hop Reasoning)
5. **Breakdown by Domain** — Per-domain metrics (e.g., Finance, Medical)
6. **Experiment Details** — Source file mapping

## Customization

The script auto-discovers all `eval_*.jsonl` files in the experiment directory. The display name for each configuration is derived from the filename by stripping the `eval_` prefix and `.jsonl` suffix (e.g., `eval_original_kg_all.jsonl` → "original_kg_all").

To add new experiments, simply place additional `eval_*.jsonl` files in the experiment directory and re-run the script.
