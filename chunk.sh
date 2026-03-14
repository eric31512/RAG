source .venv/bin/activate

python src/build_kg_from_chunks.py \
    --lang 'en' \
    --chunk_cache "cache/chunk_cache/en_contextual_chunksize512" \
    --output "graph/original_contextual_kg_all" \