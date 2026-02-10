#!/bin/bash
# Nano-GraphRAG Evaluation Script (Low Memory Version)
# Run with: tmux new-session -d -s rageval 'bash run_eval.sh'

set -e
cd /tmp2/cctsai/project/RAG

# Activate virtual environment
source .venv/bin/activate

echo "=========================================="
echo "Nano-GraphRAG + RAGEval Evaluation"
echo "Started at: $(date)"
echo "=========================================="

# Step 1: Sample queries (already done, but ensure it exists)
if [ ! -f "./dragonball_dataset/eval_queries_en.jsonl" ]; then
    echo "[Step 1] Sampling 30 English queries..."
    python3 sample_eval_queries.py --num 30 --language en
fi
echo "[Step 1] ✓ Eval queries ready"

# Step 2: Run RAG WITHOUT KG
echo ""
echo "[Step 2] Running RAG inference WITHOUT KG..."
cd My_RAG
python3 main.py \
    --query_path ../dragonball_dataset/eval_queries_en.jsonl \
    --docs_path ../dragonball_dataset/dragonball_docs.jsonl \
    --language en \
    --output ../eval_output_no_kg_en.jsonl
echo "[Step 2] ✓ RAG without KG completed"

# Step 3: Run RAG WITH KG
echo ""
echo "[Step 3] Running RAG inference WITH KG..."
python3 main.py \
    --query_path ../dragonball_dataset/eval_queries_en.jsonl \
    --docs_path ../dragonball_dataset/dragonball_docs.jsonl \
    --language en \
    --output ../eval_output_with_kg_en.jsonl \
    --use-kg
echo "[Step 3] ✓ RAG with KG completed"

# Step 4: Run RAGEval on both outputs
echo ""
echo "[Step 4] Running RAGEval evaluation..."
cd ../rageval/evaluation
python3 main.py \
    --input_file ../../eval_output_no_kg_en.jsonl \
    --output_file ../../eval_results_no_kg_en.jsonl \
    --language en \
    --num_workers 2

python3 main.py \
    --input_file ../../eval_output_with_kg_en.jsonl \
    --output_file ../../eval_results_with_kg_en.jsonl \
    --language en \
    --num_workers 2
echo "[Step 4] ✓ RAGEval completed"

# Step 5: Compare results
echo ""
echo "[Step 5] Generating comparison report..."
cd ../..
python3 compare_results.py
echo "[Step 5] ✓ Comparison report generated"

echo ""
echo "=========================================="
echo "Evaluation completed at: $(date)"
echo "Results saved to:"
echo "  - eval_results_no_kg_en.jsonl"
echo "  - eval_results_with_kg_en.jsonl" 
echo "  - comparison_report.md"
echo "=========================================="
