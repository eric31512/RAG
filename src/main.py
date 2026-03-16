from tqdm import tqdm
from pathlib import Path
from utils import load_jsonl, save_jsonl
from retriever_main import create_retriever
from chunker import recursive_chunk
from generator import generate_answer
import argparse
import os

def main(query_path, docs_path, language, output_path, mode="hybrid", kg_dir=None):
    """
    Main RAG pipeline with configurable retrieval mode.
    
    Args:
        mode: 
            - 'hybrid': BM25+Vector+Reranker only
            - 'kg': Regular KG only
            - 'kg-contextual': Contextual KG only
            - 'hybrid-kg': Hybrid + Regular KG
            - 'hybrid-kg-contextual': Hybrid + Contextual KG
    """
    # 1. Load Data
    print(f"Loading documents... (mode={mode})")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents
    print("Chunking documents...")
    if language=="zh":
        chunks = recursive_chunk(docs_for_chunking, language, chunk_size=128)
    else:
        chunks = recursive_chunk(docs_for_chunking, language, chunk_size=512)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Retriever (enable KG if mode uses it)
    print("Creating retriever...")
    chunk_size = 128 if language == "zh" else 512
    use_kg = mode in ["kg", "kg-contextual", "hybrid-kg", "hybrid-kg-contextual"]
    contextual_kg = mode in ["kg-contextual", "hybrid-kg-contextual"]
    retriever = create_retriever(chunks, language, chunksize=chunk_size, use_kg=use_kg, contextual_kg=contextual_kg, kg_dir=kg_dir)
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
        answer = generate_answer(query_text, final_chunks, language)
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
    parser.add_argument('--output', help='Path to the output file')
    parser.add_argument('--mode', 
                        choices=['hybrid', 'kg', 'kg-contextual', 'hybrid-kg', 'hybrid-kg-contextual'], 
                        default='hybrid',
                        help='Retrieval mode: hybrid, kg, kg-contextual, hybrid-kg, hybrid-kg-contextual')
    parser.add_argument('--kg_dir', help='Custom Knowledge Graph directory to load', default=None)
    args = parser.parse_args()
    
    # Map KG modes to use correct retrieve mode
    retrieve_mode = mode_mapping = {
        'hybrid': 'hybrid',
        'kg': 'kg',
        'kg-contextual': 'kg',
        'hybrid-kg': 'hybrid-kg',
        'hybrid-kg-contextual': 'hybrid-kg'
    }.get(args.mode, 'hybrid')
    
    main(args.query_path, args.docs_path, args.language, args.output, mode=args.mode, kg_dir=args.kg_dir)

