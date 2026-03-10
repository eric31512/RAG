source .venv/bin/activate

python src/build_kg_from_chunks.py \
    --lang 'en' \
    --chunk_cache "cache/chunk_cache/test_chunk.json" \
    --output "graph/merged12_kg" \
    --merge_chunks 12 --overlap 2
