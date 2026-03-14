import networkx as nx
from collections import Counter
import os

filepath = "/tmp2/cctsai/project/RAG/graph/prefix_original_kg_all/graph_chunk_entity_relation.graphml"

if not os.path.exists(filepath):
    print(f"File not found: {filepath}")
else:
    print(f"=== prefix_original_kg_all ===")
    G = nx.read_graphml(filepath)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    if len(G) > 0:
        if G.is_directed():
            cc = list(nx.weakly_connected_components(G))
        else:
            cc = list(nx.connected_components(G))
        cc.sort(key=len, reverse=True)
        print(f"Connected Components: {len(cc)}")
        print(f"Largest Component Size: {len(cc[0]) if cc else 0} nodes")
        if len(cc) > 1:
            print(f"Second Largest Component Size: {len(cc[1])} nodes")
            print(f"Third Largest Component Size: {len(cc[2])} nodes")
            
        # Count components of size 1 or 2
        tiny_components = sum(1 for c in cc if len(c) <= 3)
        print(f"Components with <= 3 nodes: {tiny_components}")
    
    entity_types = []
    for n, d in G.nodes(data=True):
        entity_types.append(d.get('entity_type', 'MISSING_TYPE'))
    type_counts = Counter(entity_types)
    
    unknown_type = '"UNKNOWN"' if '"UNKNOWN"' in type_counts else 'UNKNOWN'
    unknown_ratio = type_counts.get(unknown_type, 0) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    print(f"UNKNOWN Entity Ratio: {unknown_ratio:.2%}")
    
    degrees_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 Hub Nodes (Node, Degree):")
    for n, d in degrees_sorted[:10]:
        print(f" - {n}: {d}")
