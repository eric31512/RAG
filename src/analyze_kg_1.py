import networkx as nx
import json
from collections import Counter
import os

files = {
    "merge12_1": "/tmp2/cctsai/project/RAG/graph/merged12_kg_1/graph_chunk_entity_relation.graphml",
    "original_contextual_kg_1": "/tmp2/cctsai/project/RAG/graph/original_contextual_kg_1/graph_chunk_entity_relation.graphml",
    "original_kg_1": "/tmp2/cctsai/project/RAG/graph/original_kg_1/graph_chunk_entity_relation.graphml",
    "prefix_original_kg_1": "/tmp2/cctsai/project/RAG/graph/prefix_original_kg_1/graph_chunk_entity_relation.graphml"
}

for name, filepath in files.items():
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue

    print(f"=== {name} ===")
    try:
        G = nx.read_graphml(filepath)
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        
        # Degree info
        if G.number_of_nodes() > 0:
            degrees = [d for n, d in G.degree()]
            avg_deg = sum(degrees) / len(degrees)
            print(f"Avg Degree: {avg_deg:.2f}")
        
        # Connected components
        if len(G) > 0:
            if G.is_directed():
                cc = list(nx.weakly_connected_components(G))
            else:
                cc = list(nx.connected_components(G))
            cc.sort(key=len, reverse=True)
            print(f"Connected Components: {len(cc)}")
            print(f"Largest Component Size: {len(cc[0]) if cc else 0} nodes")
        
        # Node types (entity_type)
        entity_types = []
        for n, d in G.nodes(data=True):
            entity_types.append(d.get('entity_type', 'MISSING_TYPE'))
        type_counts = Counter(entity_types)
        print(f"Entity Types: {type_counts.most_common(5)}")
        
        unknown_type = '"UNKNOWN"' if '"UNKNOWN"' in type_counts else 'UNKNOWN'
        unknown_ratio = type_counts.get(unknown_type, 0) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        print(f"UNKNOWN Entity Ratio: {unknown_ratio:.2%}")
        
        # Top hub nodes
        degrees_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        print(f"Top 3 Hub Nodes (Node, Degree): {degrees_sorted[:3]}")

        print()
    except Exception as e:
        print(f"Error loading {name}: {e}")
        print()
