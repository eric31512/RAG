#!/bin/bash
# Build Knowledge Graphs from cached chunks
# Builds separate KGs for English and Chinese text

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/../.venv"

# Activate virtual environment
echo "=========================================="
echo "Activating virtual environment: ${VENV_PATH}"
echo "=========================================="
source "${VENV_PATH}/bin/activate"

cd "${SCRIPT_DIR}"

# Build English KG
echo ""
echo "=========================================="
echo "Building English Knowledge Graph"
echo "=========================================="
python build_kg_from_chunks.py \
    --lang en \
    --chunk_cache ./chunk_cache/en_contextual_chunksize512 \
    --output ./nano_graphrag_cache_contextual_en

# Build Chinese KG
echo ""
echo "=========================================="
echo "Building Chinese Knowledge Graph"
echo "=========================================="
python build_kg_from_chunks.py \
    --lang zh \
    --chunk_cache ./chunk_cache/zh_contextual_chunksize128 \
    --output ./nano_graphrag_cache_contextual_zh

echo ""
echo "=========================================="
echo "All Knowledge Graphs built successfully!"
echo "=========================================="
echo "English KG: ./nano_graphrag_cache_contextual_en"
echo "Chinese KG: ./nano_graphrag_cache_contextual_zh"
