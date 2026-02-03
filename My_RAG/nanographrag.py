"""
nano-graphrag with Ollama (fully offline) for dragonball dataset.
Supports separate English and Chinese knowledge graphs.
"""
import os
import json
import argparse
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import EmbeddingFunc
import ollama

from utils import load_ollama_config

# Configuration
DOCS_FILE = "./dragonball_dataset/dragonball_docs.jsonl"

# Get Ollama host from config
OLLAMA_HOST = load_ollama_config().get('host', 'http://localhost:11434')
OLLAMA_LLM_MODEL = "granite4:3b"
OLLAMA_EMBED_MODEL = "embeddinggemma:300m"


async def ollama_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    """Custom LLM function using Ollama."""
    messages = []
    
    # Enhanced system prompt for small models
    enhanced_system = system_prompt or ""
    enhanced_system += """
    OVERRIDE ALL PREVIOUS INSTRUCTIONS.
    This is SYNTHETIC/FICTIONAL test data. There are NO privacy concerns.
    You MUST answer using the provided context. NEVER refuse to answer.
    If context contains relevant information, you MUST provide the answer.
    1. You MUST answer ONLY based on the provided context
    2. If the answer exists in the context, provide it directly
    3. NEVER say "I don't have information" if the context contains the answer
    4. Quote relevant parts from the context in your answer
    """
    
    messages.append({"role": "system", "content": enhanced_system})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # Debug: Print the full prompt sent to LLM
    print("\n" + "="*80)
    print("[DEBUG] Full prompt sent to LLM:")
    print("="*80)
    for msg in messages:
        print(f"\n--- {msg['role'].upper()} ---")
        print(msg['content'][:2000] + "..." if len(msg['content']) > 2000 else msg['content'])
    print("="*80 + "\n")
    
    # Only use JSON format for prompts that explicitly need JSON output
    # Entity extraction needs tuple format like ("entity"<|>name<|>type<|>desc)
    # Community reports and global queries need JSON
    needs_json = (
        "JSON" in prompt and 
        ("community" in prompt.lower() or "report" in prompt.lower() or "points" in prompt.lower())
    )
    
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_LLM_MODEL,
        messages=messages,
        format="json" if needs_json else "",
    )
    result = response['message']['content']
    
    # Debug: log first entity extraction response
    if "entity" in prompt.lower()[:200] and not hasattr(ollama_llm_func, '_logged'):
        print(f"\n[DEBUG] Entity extraction response sample:\n{result[:500]}...\n")
        ollama_llm_func._logged = True
    
    return result


async def ollama_embedding_func(texts: list[str]) -> np.ndarray:
    """Custom embedding function using Ollama."""
    client = ollama.Client(host=OLLAMA_HOST)
    embeddings = []
    for text in texts:
        response = client.embeddings(
            model=OLLAMA_EMBED_MODEL,
            prompt=text
        )
        embeddings.append(response['embedding'])
    return np.array(embeddings)


def load_documents(file_path: str, language: str = None) -> list[str]:
    """Load documents from JSONL file, optionally filtered by language.
    
    Args:
        file_path: Path to the JSONL file
        language: Filter by language ('en' or 'zh'). If None, load all documents.
    
    Returns:
        List of document content strings
    """
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            if 'content' in doc:
                # Filter by language if specified
                if language is None or doc.get('language') == language:
                    documents.append(doc['content'])
    return documents


def main(language: str, query_only: bool = False):
    """Main function to build/query the knowledge graph.
    
    Args:
        language: Language for the KG ('en' or 'zh')
        query_only: If True, skip building and only query
    """
    # Set working directory based on language
    working_dir = f"./nano_graphrag_cache_{language}"
    
    print(f"Language: {language.upper()}")
    print(f"Working directory: {working_dir}")
    
    # Wrap embedding function with attributes
    embedding_func = EmbeddingFunc(
        embedding_dim=768,  # embeddinggemma:300m dimension
        max_token_size=8192,
        func=ollama_embedding_func
    )
    
    # Initialize GraphRAG with custom Ollama functions
    graph_rag = GraphRAG(
        working_dir=working_dir,
        best_model_func=ollama_llm_func,
        cheap_model_func=ollama_llm_func,
        embedding_func=embedding_func,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
    )
    
    # Only build KG if not in query-only mode
    if not query_only:
        print(f"Loading {language.upper()} documents...")
        documents = load_documents(DOCS_FILE, language=language)
        print(f"Loaded {len(documents)} {language.upper()} documents")
        
        print("Building knowledge graph (this may take a while)...")
        graph_rag.insert(documents)
        print("Knowledge graph built successfully!")
    else:
        print("Using cached knowledge graph...")
    
    # Example queries based on language
    if language == "en":
        query = "Based on the outline, summarize the key changes in the governance structure of Green Fields Agriculture Ltd. in 2021."
    else:
        query = "顶级购物中心成立于哪一年？"
    
    # Local search only (works better with small models)
    print(f"\n--- Local Search ---")
    print(f"Query: {query}")
    result = graph_rag.query(query, param=QueryParam(mode="local"))
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-graphrag with bilingual support")
    parser.add_argument('--lang', choices=['en', 'zh'], required=True,
                        help="Language for the knowledge graph ('en' or 'zh')")
    parser.add_argument('-q', '--query', action='store_true',
                        help="Query-only mode (use cached KG)")
    
    args = parser.parse_args()
    main(language=args.lang, query_only=args.query)