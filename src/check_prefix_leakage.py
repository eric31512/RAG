import json
import networkx as nx
import os
import re

def parse_chunk_content(content):
    """
    Splits the chunk content into background context and target chunk.
    Returns (background_text, target_text).
    """
    bg_marker = "-Background Context (DO NOT extract from this section)-"
    target_marker = "-Target Chunk (Extract ONLY from the text below)-"
    
    if bg_marker in content and target_marker in content:
        parts = content.split(target_marker)
        bg_text = parts[0].replace(bg_marker, "").strip()
        target_text = parts[1].strip()
        return bg_text, target_text
    return "", content

def check_leakage(graph_path, chunks_path):
    if not os.path.exists(graph_path) or not os.path.exists(chunks_path):
        print("Files not found.")
        return
        
    G = nx.read_graphml(graph_path)
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
        
    total_entity_chunk_pairs = 0
    in_target = 0
    in_prefix_only = 0
    hallucinated = 0
    
    leaked_entities = set()
    
    # Analyze each node's sources
    for node, data in G.nodes(data=True):
        source_id_str = data.get('source_id', '')
        if not source_id_str:
            continue
            
        source_ids = source_id_str.split("<SEP>")
        
        for chunk_id in source_ids:
            if chunk_id not in chunks_data:
                continue
                
            chunk_content = chunks_data[chunk_id].get('content', '')
            bg_text, target_text = parse_chunk_content(chunk_content)
            
            total_entity_chunk_pairs += 1
            
            # Simple substring matching (case-insensitive)
            # In a real scenario, you might want more robust un-escaping or tokenization,
            # but simple lowercase exact match works for general diagnostics.
            node_lower = str(node).lower().strip()
            # remove surrounding quotes if any
            if node_lower.startswith('"') and node_lower.endswith('"'):
                node_lower = node_lower[1:-1]
                
            in_t = node_lower in target_text.lower()
            in_p = node_lower in bg_text.lower()
            
            if in_t:
                in_target += 1
            elif in_p:
                in_prefix_only += 1
                leaked_entities.add(node)
                # print(f"Leak detected: '{node}' in {chunk_id}")
            else:
                hallucinated += 1
                
    print(f"--- Analysis Report ---")
    print(f"Total Entity-Chunk Extractions: {total_entity_chunk_pairs}")
    print(f"Valid Extractions (found in Target Chunk): {in_target} ({(in_target/total_entity_chunk_pairs)*100:.2f}%)")
    print(f"Prefix Leakage (found ONLY in Background Context): {in_prefix_only} ({(in_prefix_only/total_entity_chunk_pairs)*100:.2f}%)")
    print(f"Hallucinated (found in neither): {hallucinated} ({(hallucinated/total_entity_chunk_pairs)*100:.2f}%)")
    print(f"Total Unique Nodes experiencing Prefix Leakage: {len(leaked_entities)}")
    
    print("\nTop 10 Leaked Entities (by degree in graph):")
    # Sort leaked entities by their degree in the graph to see which ones are prominent
    leaked_with_degree = [(n, G.degree[n]) for n in leaked_entities if n in G]
    leaked_with_degree.sort(key=lambda x: x[1], reverse=True)
    for n, d in leaked_with_degree[:10]:
        print(f" - {n} (Degree: {d})")

if __name__ == "__main__":
    KG_DIR = "/tmp2/cctsai/project/RAG/graph/prefix_original_kg_1"
    graphml_file = os.path.join(KG_DIR, "graph_chunk_entity_relation.graphml")
    chunks_file = os.path.join(KG_DIR, "kv_store_text_chunks.json")
    
    check_leakage(graphml_file, chunks_file)
