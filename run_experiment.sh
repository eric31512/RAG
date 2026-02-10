#!/bin/bash
# =============================================================================
# Multi-Mode RAG Retrieval Experiment
# 測試三種檢索模式: hybrid, kg, hybrid-kg
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

# Paths
QUERY_PATH="dragonball_dataset/eval_queries_en.jsonl"
DOCS_PATH="dragonball_dataset/dragonball_docs.jsonl"
LANGUAGE="en"

# Output directories
OUTPUT_DIR="experiment_outputs"
mkdir -p "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/run_$TIMESTAMP"
mkdir -p "$RUN_DIR"

# Log file
LOG_FILE="$RUN_DIR/experiment.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=============================================="
log "Multi-Mode RAG Experiment"
log "Run ID: $TIMESTAMP"
log "Log file: $LOG_FILE"
log "=============================================="

# =============================================================================
# Step 1: Run RAG inference for each mode
# =============================================================================

# All 5 retrieval modes to test
MODES=("hybrid" "kg" "kg-contextual" "hybrid-kg" "hybrid-kg-contextual")

for MODE in "${MODES[@]}"; do
    log ""
    log "[Step 1] Running RAG inference with mode: $MODE"
    log "----------------------------------------------"
    
    OUTPUT_FILE="$RUN_DIR/output_${MODE}.jsonl"
    
    cd My_RAG
    python3 main.py \
        --query_path "../$QUERY_PATH" \
        --docs_path "../$DOCS_PATH" \
        --language "$LANGUAGE" \
        --output "../$OUTPUT_FILE" \
        --mode "$MODE" 2>&1 | tee -a "../$LOG_FILE"
    cd ..
    
    log "Saved: $OUTPUT_FILE"
done

# =============================================================================
# Step 2: Run RAGEval on each output
# =============================================================================

log ""
log "[Step 2] Running RAGEval evaluations..."
log "----------------------------------------------"

for MODE in "${MODES[@]}"; do
    log "Evaluating: $MODE"
    
    INPUT_FILE="$RUN_DIR/output_${MODE}.jsonl"
    RESULT_FILE="$RUN_DIR/eval_${MODE}.jsonl"
    
    cd rageval/evaluation
    python3 main.py \
        --input_file "../../$INPUT_FILE" \
        --output_file "../../$RESULT_FILE" \
        --language "$LANGUAGE" \
        --num_workers 2 \
        --model "llama3.1:8b" 2>&1 | tee -a "../../$LOG_FILE"
    cd ../..
    
    log "Saved: $RESULT_FILE"
done

# =============================================================================
# Step 3: Generate comparison report
# =============================================================================

log ""
log "[Step 3] Generating comparison report..."
log "----------------------------------------------"

REPORT_FILE="$RUN_DIR/comparison_report.md"

python3 << EOF
import json
from pathlib import Path
from collections import defaultdict

run_dir = Path("$RUN_DIR")
modes = ["hybrid", "kg", "kg-contextual", "hybrid-kg", "hybrid-kg-contextual"]

# Collect results
results = {}
for mode in modes:
    eval_file = run_dir / f"eval_{mode}.jsonl"
    if not eval_file.exists():
        print(f"Warning: {eval_file} not found")
        continue
    
    scores = defaultdict(list)
    with open(eval_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            for key, value in data.items():
                # Capture all numeric metrics
                if isinstance(value, (int, float)):
                    if any(k in key.lower() for k in ['score', 'precision', 'recall', 'completeness', 'hallucination', 'irrelevance']):
                        scores[key].append(value)
    
    # Calculate averages
    results[mode] = {k: sum(v)/len(v) if v else 0 for k, v in scores.items()}

# Generate markdown report
report_lines = [
    "# Multi-Mode RAG Experiment Report",
    "",
    f"**Run ID**: $TIMESTAMP",
    f"**Language**: $LANGUAGE",
    "",
    "## Results Summary",
    "",
    "| Metric | Hybrid | KG | KG-Ctx | Hybrid+KG | Hybrid+KG-Ctx |",
    "|--------|--------|-------|--------|-----------|---------------|",
]

# Get all metrics
all_metrics = set()
for mode_results in results.values():
    all_metrics.update(mode_results.keys())

def fmt(val):
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val) if val != "-" else "-"

for metric in sorted(all_metrics):
    vals = [results.get(m, {}).get(metric, "-") for m in modes]
    row = f"| {metric} | " + " | ".join(fmt(v) for v in vals) + " |"
    report_lines.append(row)

report_lines.extend([
    "",
    "## Mode Descriptions",
    "",
    "| Mode | Description |",
    "|------|-------------|",
    "| **Hybrid** | BM25 + Vector (Dense) + Reranker |",
    "| **KG** | Regular KG retrieval only |",
    "| **KG-Ctx** | Contextual KG retrieval only |",
    "| **Hybrid+KG** | Hybrid retrieval + Regular KG context |",
    "| **Hybrid+KG-Ctx** | Hybrid retrieval + Contextual KG context |",
    "",
    "## Files",
    "",
])

for mode in modes:
    report_lines.append(f"- \`output_{mode}.jsonl\`: RAG predictions")
    report_lines.append(f"- \`eval_{mode}.jsonl\`: Evaluation results")

with open(run_dir / "comparison_report.md", "w") as f:
    f.write("\\n".join(report_lines))

print(f"Report saved: {run_dir / 'comparison_report.md'}")
EOF

# =============================================================================
# Summary
# =============================================================================

log ""
log "=============================================="
log "Experiment Complete!"
log "=============================================="
log ""
log "Results saved to: $RUN_DIR"
log ""
log "Files:"
ls -la "$RUN_DIR" | tee -a "$LOG_FILE"
log ""
log "View report: cat $REPORT_FILE"
log "View logs: cat $LOG_FILE"
