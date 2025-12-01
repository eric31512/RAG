from tqdm import tqdm
from pathlib import Path
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever
# from denseRetriever import create_retriever 
from generator import generate_answer
import argparse
from query_rewriter import rewrite_query

def main(query_path, docs_path, language, output_path):
    # 1. Load Data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents
    print("Chunking documents...")
    precomputed_path = Path(__file__).parent.parent / "precomputed" / f"precomputed_chunks_{language}.jsonl"
    if precomputed_path.exists():
        print(f"Using precomputed chunks from {precomputed_path}")
        chunks = load_jsonl(precomputed_path)
    else:
        print("No precomputed chunks found, running chunk_documents at runtime.")
        chunks = chunk_documents(docs_for_chunking, language)
        
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Retriever
    print("Creating retriever...")
    retriever = create_retriever(chunks, language)
    print("Retriever created successfully.")


    for query in tqdm(queries, desc="Processing Queries"):
        # 4. Retrieve relevant chunks
        query_text = query["query"]["content"]

        # choose mode: "none" / "multi" / "hyde" / "decompose" / "stepback"
        rewritten_queries = rewrite_query(
            query_text,
            language=language,
            mode="stepback",      # or "multi", "hyde", "decompose", "stepback" ,"none"
            num_queries=3      # optional, for "multi"
        )

        FINAL_TOP_K = 5

        all_chunks = []
        for q in rewritten_queries:
            retrieved = retriever.retrieve(q, top_k=5)
            all_chunks.extend(retrieved)

        # Deduplicate by retriever_id
        unique = {}
        for c in all_chunks:
            meta = c.get("metadata", {})
            key = meta.get("retriever_id")
            if key is None:
                key = id(c)
            if key not in unique:
                unique[key] = c

        final_chunks = list(unique.values())[:FINAL_TOP_K]

        # 5. Generate Answer
        print("Generating answer...")
        answer = generate_answer(query_text, final_chunks, language)
        print(f"Generated Answer: {answer}") #for test prompt

        query["prediction"]["content"] = answer
        if final_chunks:
            query["prediction"]["references"] = [c["page_content"] for c in final_chunks]   
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
    args = parser.parse_args()
    main(args.query_path, args.docs_path, args.language, args.output)
