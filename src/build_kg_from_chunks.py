"""
Build nano-graphrag KG using pre-cached contextual chunks.
Uses chunk_cache instead of re-chunking documents.
"""
import os
import json
import argparse
from collections import defaultdict
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import EmbeddingFunc
from nano_graphrag.prompt import PROMPTS
import ollama

from utils import load_ollama_config

# Configuration
OLLAMA_HOST = load_ollama_config().get('host', 'http://localhost:11434')
OLLAMA_LLM_MODEL = "granite4:3b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"  # Working Ollama embedding model



# ===== Prompt 選擇 =====
# 帶 prefix 版本：解耦 Context 與 Extraction
PROMPT_WITH_PREFIX = """-Goal-
Given a text document containing a Background Context and a Target Chunk, 
identify all entities and relationships from the TARGET CHUNK ONLY.
The Background Context is provided STRICTLY for understanding the situation (e.g., resolving pronouns).

-Steps-
1. Identify all entities from the Target Chunk. For each identified entity, extract:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity  
- relationship_description: explanation of why they are related
- relationship_strength: a numeric score indicating strength of the relationship
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output as a single list using **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

-Strict Rules-
- NEVER extract any entity that appears ONLY in the Background Context.
- Use the Background Context ONLY to resolve ambiguous pronouns (like "it", "they", "the company") found in the Target Chunk. Extract the resolved proper name if the pronoun points to it.
- Focus strictly on specific actors, metrics, events, and their direct causal relationships within the Target Chunk.
- Never invent entities not explicitly present in the Target Chunk.
- If an entity appears in the Background Context AND the Target Chunk, you MUST extract it.

######################
-Example-

Entity_types: [organization, person, event, geo, metric]

Input:
-Background Context (DO NOT extract from this section)-
This section describes Acme Government Solutions' decision to distribute dividends to shareholders following a major government contract acquisition. The contract was originally negotiated by their partner, Beta Technologies.

-Target Chunk (Extract ONLY from the text below)-
In January 2021, Acme Government Solutions made a significant decision to distribute $5 million of dividends to its shareholders. The company also announced a new $100 million government contract.

Output:
("entity"{tuple_delimiter}ACME GOVERNMENT SOLUTIONS{tuple_delimiter}ORGANIZATION{tuple_delimiter}Acme Government Solutions is a government services company that distributed dividends and acquired a major government contract.){record_delimiter}
("entity"{tuple_delimiter}DIVIDEND DISTRIBUTION{tuple_delimiter}EVENT{tuple_delimiter}In January 2021, Acme Government Solutions distributed $5 million of dividends to its shareholders.){record_delimiter}
("entity"{tuple_delimiter}GOVERNMENT CONTRACT{tuple_delimiter}EVENT{tuple_delimiter}Acme Government Solutions announced a new $100 million government contract.){record_delimiter}
("entity"{tuple_delimiter}$5 MILLION{tuple_delimiter}METRIC{tuple_delimiter}The amount of dividends distributed to shareholders by Acme Government Solutions in January 2021.){record_delimiter}
("entity"{tuple_delimiter}$100 MILLION{tuple_delimiter}METRIC{tuple_delimiter}The value of the new government contract announced by Acme Government Solutions.){record_delimiter}
("relationship"{tuple_delimiter}ACME GOVERNMENT SOLUTIONS{tuple_delimiter}DIVIDEND DISTRIBUTION{tuple_delimiter}Acme Government Solutions made the decision to distribute dividends to shareholders.{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}ACME GOVERNMENT SOLUTIONS{tuple_delimiter}GOVERNMENT CONTRACT{tuple_delimiter}Acme Government Solutions announced a new government contract.{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}DIVIDEND DISTRIBUTION{tuple_delimiter}$5 MILLION{tuple_delimiter}The dividend distribution amounted to $5 million.{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}GOVERNMENT CONTRACT{tuple_delimiter}$100 MILLION{tuple_delimiter}The announced government contract was worth $100 million.{tuple_delimiter}10)
{completion_delimiter}

Note: 
1. "ACME GOVERNMENT SOLUTIONS" appeared in both the Context and the Chunk. It was correctly extracted.
2. "BETA TECHNOLOGIES" appeared in the Background Context but NOT in the Target Chunk. It was correctly IGNORED and NOT extracted.
3. "The company" in the Chunk was correctly resolved to "ACME GOVERNMENT SOLUTIONS" using the Context.
######################

######################
-Real Data-
Entity_types: {entity_types}

Input:
{input_text}
######################
Output:
"""

# 不帶 prefix 版本：同樣的 prompt 結構，但只處理 chunk 本身
PROMPT_WITHOUT_PREFIX = """-Goal-
Given a text document, identify all entities and relationships from the text.

-Steps-
1. Identify all entities from the text. For each identified entity, extract:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity  
- relationship_description: explanation of why they are related
- relationship_strength: a numeric score indicating strength of the relationship
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output as a single list using **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

-Strict Rules-
- Focus on specific actors, metrics, events, and their direct causal relationships within the text.
- Never invent entities not explicitly present in the text.

######################
-Example-

Entity_types: [organization, person, event, geo, metric]

Input:
In January 2021, Acme Government Solutions made a significant decision to distribute $5 million of dividends to its shareholders. This dividend distribution was a result of the company's successful acquisition of a major government contract worth $100 million in March 2021.

Output:
("entity"{tuple_delimiter}ACME GOVERNMENT SOLUTIONS{tuple_delimiter}ORGANIZATION{tuple_delimiter}Acme Government Solutions is a government services company that distributed dividends and acquired a major government contract.){record_delimiter}
("entity"{tuple_delimiter}DIVIDEND DISTRIBUTION{tuple_delimiter}EVENT{tuple_delimiter}In January 2021, Acme Government Solutions distributed $5 million of dividends to its shareholders.){record_delimiter}
("entity"{tuple_delimiter}GOVERNMENT CONTRACT ACQUISITION{tuple_delimiter}EVENT{tuple_delimiter}In March 2021, Acme Government Solutions acquired a major government contract worth $100 million.){record_delimiter}
("entity"{tuple_delimiter}$5 MILLION{tuple_delimiter}METRIC{tuple_delimiter}The amount of dividends distributed to shareholders by Acme Government Solutions in January 2021.){record_delimiter}
("entity"{tuple_delimiter}$100 MILLION{tuple_delimiter}METRIC{tuple_delimiter}The value of the major government contract acquired by Acme Government Solutions in March 2021.){record_delimiter}
("relationship"{tuple_delimiter}ACME GOVERNMENT SOLUTIONS{tuple_delimiter}DIVIDEND DISTRIBUTION{tuple_delimiter}Acme Government Solutions made the decision to distribute dividends to shareholders.{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}ACME GOVERNMENT SOLUTIONS{tuple_delimiter}GOVERNMENT CONTRACT ACQUISITION{tuple_delimiter}Acme Government Solutions successfully acquired a major government contract.{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}GOVERNMENT CONTRACT ACQUISITION{tuple_delimiter}DIVIDEND DISTRIBUTION{tuple_delimiter}The dividend distribution was a direct result of the successful government contract acquisition.{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}DIVIDEND DISTRIBUTION{tuple_delimiter}$5 MILLION{tuple_delimiter}The dividend distribution amounted to $5 million.{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}GOVERNMENT CONTRACT ACQUISITION{tuple_delimiter}$100 MILLION{tuple_delimiter}The acquired government contract was worth $100 million.{tuple_delimiter}10)
{completion_delimiter}
######################

######################
-Real Data-
Entity_types: {entity_types}

Input:
{input_text}
######################
Output:
"""


async def ollama_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    """Custom LLM function using Ollama."""
    messages = []
    
    enhanced_system = system_prompt or ""
        
    messages.append({"role": "system", "content": enhanced_system})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
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
    return response['message']['content']


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

def load_cached_chunks(cache_path: str, use_prefix: bool = True) -> list[str]:
    """Load pre-cached chunks.
    
    Args:
        cache_path: Path to the chunk cache file (JSON)
        use_prefix: If True, include contextual prefix as Background Context.
                    If False, only use the original content.
    """
    with open(cache_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    documents = []
    for chunk in chunks:
        meta = chunk.get('metadata', {})
        content = meta.get('original_content', chunk['page_content'])
        
        if use_prefix:
            prefix = meta.get('contextual_prefix', '')
            # 用明確標記讓 LLM 區分背景與目標
            combined = (
                f"-Background Context (DO NOT extract from this section)-\n"
                f"{prefix}\n\n"
                f"-Target Chunk (Extract ONLY from the text below)-\n"
                f"{content}"
            )
        else:
            combined = content
        
        documents.append(combined)
    
    print(f"Loaded {len(documents)} chunks (use_prefix={use_prefix}) from cache")
    return documents

#merge prefix and content
# def load_cached_chunks(cache_path: str , use_prefix: bool = True) -> list[str]:
#     """Load pre-cached chunks and return as list of strings.
    
#     Args:
#         cache_path: Path to the chunk cache file (JSON)
    
#     Returns:
#         List of chunk content strings
#     """
#     with open(cache_path, 'r', encoding='utf-8') as f:
#         chunks = json.load(f)
    
#     # Extract page_content from each chunk
#     documents = [chunk['page_content'] for chunk in chunks]
#     print(f"Loaded {len(documents)} chunks from cache")
#     return documents

# merge 12
def load_merged_chunks(cache_path: str, group_size: int = 12, overlap: int = 2) -> list[str]:
    """Load chunks and merge N adjacent chunks per doc_id with sliding window overlap.
    
    Args:
        cache_path: Path to the chunk cache file (JSON)
        group_size: Number of adjacent chunks to merge (default: 12)
        overlap: Number of chunks to overlap between windows (default: 2)
    
    Returns:
        List of merged chunk strings
    """
    with open(cache_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Group chunks by doc_id, preserving order
    doc_chunks = defaultdict(list)
    for chunk in chunks:
        doc_id = chunk.get('metadata', {}).get('doc_id', 'unknown')
        content = chunk.get('metadata', {}).get('original_content', chunk['page_content'])
        doc_chunks[doc_id].append(content)
    
    # Merge with sliding window: stride = group_size - overlap
    stride = group_size - overlap
    documents = []
    for doc_id, contents in doc_chunks.items():
        for i in range(0, len(contents), stride):
            group = contents[i:i + group_size]
            if len(group) < group_size and i > 0:
                # Skip the last window if it's too small (already covered by previous window)
                break
            merged = "\n\n".join(group)
            documents.append(merged)
    
    total_original = sum(len(v) for v in doc_chunks.values())
    print(f"Merged {total_original} chunks into {len(documents)} documents "
          f"(group_size={group_size}, overlap={overlap}, stride={stride}, doc_ids={list(doc_chunks.keys())})")
    return documents


def main(language: str, chunk_cache: str, output_dir: str = None, query_only: bool = False, 
         use_prefix: bool = True, merge_chunks: int = 1, overlap: int = 2):
    """Main function to build/query the KG from cached chunks.
    
    Args:
        language: Language code ('en' or 'zh')
        chunk_cache: Path to the chunk cache file
        output_dir: Output directory for KG cache (default: nano_graphrag_cache_contextual_{lang})
        query_only: If True, skip building and only query
        use_prefix: If True, use prefix prompt + prefix data; if False, use no-prefix prompt + raw chunks
        merge_chunks: Number of adjacent chunks to merge before extraction (1 = no merge)
    """
    # 根據 use_prefix 選擇對應的 prompt
    if use_prefix:
        PROMPTS["entity_extraction"] = PROMPT_WITH_PREFIX
        print("Using prompt: WITH prefix (decoupled context)")
    else:
        PROMPTS["entity_extraction"] = PROMPT_WITHOUT_PREFIX
        print("Using prompt: WITHOUT prefix (chunk only)")
    # Set output directory
    if output_dir is None:
        output_dir = f"./nano_graphrag_cache_contextual_{language}"
    
    print(f"Language: {language.upper()}")
    print(f"Chunk cache: {chunk_cache}")
    print(f"Output directory: {output_dir}")
    
    # Wrap embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=768,  # nomic-embed-text dimension
        max_token_size=8192,
        func=ollama_embedding_func
    )
    
    # Initialize GraphRAG
    # Use very large chunk sizes so nano-graphrag treats each input as a single chunk
    graph_rag = GraphRAG(
        working_dir=output_dir,
        best_model_func=ollama_llm_func,
        cheap_model_func=ollama_llm_func,
        embedding_func=embedding_func,
        chunk_token_size=10000 * merge_chunks,  # Scale up to prevent re-chunking merged docs
        chunk_overlap_token_size=0,
    )
    
    if not query_only:
        print(f"Loading cached chunks from: {chunk_cache}")
        if merge_chunks > 1:
            documents = load_merged_chunks(chunk_cache, group_size=merge_chunks, overlap=overlap)
        else:
            documents = load_cached_chunks(chunk_cache, use_prefix=use_prefix)
        
        print(f"Building knowledge graph from {len(documents)} chunks...")
        print("This may take a while...")
        
        # Insert chunks as documents (each chunk = one document)
        graph_rag.insert(documents)
        print("Knowledge graph built successfully!")
    else:
        print("Using cached knowledge graph...")
    
    # Test query
    if language == "en":
        query = "What governance changes happened at Green Fields Agriculture in 2021?"
    else:
        query = "顶级购物中心成立于哪一年？"
    
    print(f"\n--- Test Query ---")
    print(f"Query: {query}")
    result = graph_rag.query(query, param=QueryParam(mode="local"))
    print(f"Result: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KG from cached chunks")
    parser.add_argument('--lang', choices=['en', 'zh'], required=True,
                        help="Language for the knowledge graph")
    parser.add_argument('--chunk_cache', required=True,
                        help="Path to the chunk cache file (e.g., chunk_cache/en_contextual_chunksize512)")
    parser.add_argument('--output', default=None,
                        help="Output directory for KG cache")
    parser.add_argument('-q', '--query', action='store_true',
                        help="Query-only mode (use existing KG)")
    parser.add_argument('--use_prefix', action='store_true', default=False,
                        help="Use prefix (background context) in prompt and data (default: False)")
    parser.add_argument('--merge_chunks', type=int, default=1,
                        help="Merge N adjacent chunks per doc_id before extraction (default: 1, no merge)")
    parser.add_argument('--overlap', type=int, default=2,
                        help="Number of chunks to overlap between merged windows (default: 2)")
    
    args = parser.parse_args()
    main(language=args.lang, chunk_cache=args.chunk_cache, 
         output_dir=args.output, query_only=args.query, use_prefix=args.use_prefix,
         merge_chunks=args.merge_chunks, overlap=args.overlap)
