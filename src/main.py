from tqdm import tqdm
from pathlib import Path
from utils import load_jsonl, save_jsonl
from retriever_main import retriever
from chunker import chunker
from generator import generator
import argparse
import os

def main(query_path, docs_path, language, chunk_cache, output_path, mode="hybrid", kg_dir=None):
    """
    Main RAG pipeline with configurable retrieval mode.
    
    Args:
        mode: 
            - 'hybrid': BM25+Vector+Reranker only
            - 'kg': Regular KG only
            - 'hybrid-kg': Hybrid + Regular KG
    """
    # 1. Load Data
    print(f"Loading documents... (mode={mode})")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents
    if os.path.exists(chunk_cache):
        with open(chunk_cache, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Chunk cache hit: {chunk_cache}")
    else:
        print("Chunking documents...")
        if language=="zh":
            chunks = chunker(docs_for_chunking, language, chunk_size=128)
        else:
            chunks = chunker(docs_for_chunking, language, chunk_size=512)
        print(f"Created {len(chunks)} chunks.")

    # 3. Create Retriever (enable KG if mode uses it)
    print("Creating retriever...")
    chunk_size = 128 if language == "zh" else 512
    use_kg = mode in ["kg", "hybrid-kg"]
    retriever = retriever(chunks, language, chunksize=chunk_size, use_kg=use_kg, kg_dir=kg_dir)
    print("Retriever created successfully.")

    for query in tqdm(queries, desc="Processing Queries"):
        # 4. Retrieve relevant chunks
        query_text = query["query"]["content"]
        if language == "zh":
            FINAL_TOP_K = 3
        elif language == "en":
            FINAL_TOP_K = 3

        
        # Use mode parameter for retrieval
        retrieved = retriever.retrieve(query_text, top_k=FINAL_TOP_K, mode=mode)
        all_chunks = retrieved

        # Deduplicate by retriever_id
        seen, unique = set(), []
        for c in all_chunks:
            key = c.get("metadata", {}).get("id") or id(c)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        # retrieved_chunks = unique
        final_chunks = unique[:FINAL_TOP_K]
        
        
        # 5. Generate Answer
        print("Generating answer...")
        answer = generator(query_text, final_chunks, language)
        print(f"Generated Answer: {answer}") #for test prompt

        query["prediction"]["content"] = answer
        if final_chunks:
            query["prediction"]["references"] = [c["metadata"].get("original_content", c["page_content"]) for c in final_chunks]   
        else:
            query["prediction"]["references"] = []  

    save_jsonl(output_path, queries)
    print("Predictions saved at '{}'".format(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help='Path to the query file')
    parser.add_argument('--docs_path', help='Path to the documents file')
    parser.add_argument('--language', help='Language to filter queries (zh or en), if not specified, process all')
    parser.add_argument('--chunk_cache', type=str, default=None, help='Path to the chunk cache file')
    parser.add_argument('--output', help='Path to the output file')
    parser.add_argument('--mode', 
                        choices=['hybrid', 'kg', 'hybrid-kg'], 
                        default='hybrid',
                        help='Retrieval mode: hybrid, kg, hybrid-kg')
    parser.add_argument('--kg_dir', help='Custom Knowledge Graph directory to load', default=None)
    args = parser.parse_args()
    
    # Map KG modes to use correct retrieve mode
    retrieve_mode = mode_mapping = {
        'hybrid': 'hybrid',
        'kg': 'kg',
        'hybrid-kg': 'hybrid-kg'
    }.get(args.mode, 'hybrid')
    
    main(args.query_path, args.docs_path, args.language, args.chunk_cache, args.output, mode=args.mode, kg_dir=args.kg_dir)

