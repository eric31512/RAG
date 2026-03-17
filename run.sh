#!/bin/bash
source .venv/bin/activate
set -e

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
EXP_DIR="./experiment_outputs/run_${TIMESTAMP}"
mkdir -p "$EXP_DIR"

log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local message="$timestamp - $1"
    local len=${#message}
    local border=$(printf '=%.0s' $(seq 1 $len))
    
    echo "$border"
    echo "$message"
    echo "$border"
}

run_kg_results() {
    local language=$1
    local kg_dir=$2
    local kg_name=$(basename "$kg_dir")

    log "[INFO] Running purely KG inference for language: ${language} on ${kg_name}"
    python ./src/main.py \
        --query_path ./dragonball_dataset/eval_queries_en.jsonl \
        --docs_path ./dragonball_dataset/dragonball_docs.jsonl \
        --language ${language} \
        --output "${EXP_DIR}/output_${kg_name}.jsonl" \
        --mode kg \
        --kg_dir ${kg_dir}

    log "[INFO] Running Evaluation for language: ${language} on ${kg_name}"
    
    # Store current path and move to evaluation dir
    pushd ./rageval/evaluation > /dev/null
    
    python main.py \
        --input_file "../../${EXP_DIR}/output_${kg_name}.jsonl" \
        --output_file "../../${EXP_DIR}/eval_${kg_name}.jsonl" \
        --num_workers 5 \
        --language ${language} \
        --model "llama3.1:8b"
        
    popd > /dev/null
    
    log "[INFO] Evaluation completed. Results saved to ${EXP_DIR}/eval_${kg_name}.jsonl"
}

run_kg_results "en" "./graph/original_kg_all"
run_kg_results "en" "./graph/merged12_kg_all"

log "[INFO] All inference tasks completed."
