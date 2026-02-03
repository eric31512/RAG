"""
Knowledge Graph Visualization Script
Visualize the GraphML knowledge graph using pyvis
"""

import networkx as nx
from pyvis.network import Network
import random

def get_color_by_type(entity_type: str) -> str:
    """Assign colors based on entity type"""
    color_map = {
        "ORGANIZATION": "#FF6B6B",  # Red
        "PERSON": "#4ECDC4",         # Teal
        "LOCATION": "#45B7D1",       # Blue
        "GEO": "#45B7D1",            # Blue
        "EVENT": "#96CEB4",          # Green
        "CONCEPT": "#FFEAA7",        # Yellow
        "TECHNOLOGY": "#DDA0DD",     # Plum
        "ROLE": "#F39C12",           # Orange
        "POLICY": "#9B59B6",         # Purple
    }
    # Handle multiple types separated by comma
    if "," in entity_type:
        entity_type = entity_type.split(",")[0].strip()
    
    # Clean up the type string
    entity_type = entity_type.strip('"').upper()
    
    return color_map.get(entity_type, "#CCCCCC")  # Default gray

def visualize_kg(graphml_path: str, output_html: str = "kg_visualization.html", 
                 max_nodes: int = 100, filter_by_type: list = None):
    """
    Visualize a GraphML knowledge graph
    
    Args:
        graphml_path: Path to the GraphML file
        output_html: Output HTML file name
        max_nodes: Maximum number of nodes to display (for performance)
        filter_by_type: List of entity types to include (None = all)
    """
    print(f"Loading graph from {graphml_path}...")
    G = nx.read_graphml(graphml_path)
    
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create a subgraph if too large
    if G.number_of_nodes() > max_nodes:
        print(f"Graph too large, sampling {max_nodes} nodes...")
        # Get nodes with highest degree (most connected)
        node_degrees = dict(G.degree())
        top_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()
        print(f"Subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Filter by entity type if specified
    if filter_by_type:
        filter_by_type = [t.upper() for t in filter_by_type]
        nodes_to_keep = []
        for node in G.nodes():
            entity_type = G.nodes[node].get('entity_type', '').strip('"').upper()
            if any(t in entity_type for t in filter_by_type):
                nodes_to_keep.append(node)
        G = G.subgraph(nodes_to_keep).copy()
        print(f"Filtered graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create pyvis network
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=False,
        notebook=False
    )
    
    # Configure physics
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
                "enabled": true,
                "iterations": 100
            }
        },
        "nodes": {
            "font": {
                "size": 12,
                "face": "arial"
            }
        },
        "edges": {
            "smooth": {
                "type": "continuous"
            },
            "color": {
                "inherit": "both"
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)
    
    # Add nodes
    for node in G.nodes():
        node_data = G.nodes[node]
        entity_type = node_data.get('entity_type', 'UNKNOWN')
        description = node_data.get('description', 'No description')
        
        # Clean up node label
        label = node.strip('"')
        if len(label) > 30:
            label = label[:27] + "..."
        
        # Clean up description for tooltip
        if len(description) > 300:
            description = description[:297] + "..."
        
        color = get_color_by_type(entity_type)
        
        # Node size based on degree
        degree = G.degree(node)
        size = min(10 + degree * 2, 40)
        
        net.add_node(
            node,
            label=label,
            title=f"<b>{node.strip('\"')}</b><br><br><i>Type:</i> {entity_type}<br><br><i>Description:</i><br>{description}",
            color=color,
            size=size
        )
    
    # Add edges
    for source, target in G.edges():
        edge_data = G.edges[source, target]
        weight = float(edge_data.get('weight', 1.0))
        description = edge_data.get('description', '')
        
        net.add_edge(
            source,
            target,
            title=description,
            width=max(1, min(weight / 5, 5))  # Normalize width
        )
    
    # Save
    net.save_graph(output_html)
    print(f"✅ Visualization saved to {output_html}")
    print(f"\nOpen the file in a browser to view the interactive graph!")

def print_stats(graphml_path: str):
    """Print statistics about the knowledge graph"""
    G = nx.read_graphml(graphml_path)
    
    print("\n" + "="*50)
    print("📊 Knowledge Graph Statistics")
    print("="*50)
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Total Edges: {G.number_of_edges()}")
    
    # Count entity types
    type_counts = {}
    for node in G.nodes():
        entity_type = G.nodes[node].get('entity_type', 'UNKNOWN').strip('"')
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
    
    print("\n📌 Entity Types:")
    for entity_type, count in sorted(type_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {entity_type}: {count}")
    
    # Top connected nodes
    print("\n🔗 Top 10 Most Connected Nodes:")
    node_degrees = dict(G.degree())
    for node, degree in sorted(node_degrees.items(), key=lambda x: -x[1])[:10]:
        label = node.strip('"')
        if len(label) > 40:
            label = label[:37] + "..."
        print(f"  {label}: {degree} connections")
    
    print("="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize GraphML Knowledge Graph")
    parser.add_argument("--graphml", default="nano_graphrag_cache_en/graph_chunk_entity_relation.graphml",
                        help="Path to GraphML file")
    parser.add_argument("--output", default="kg_visualization_en.html",
                        help="Output HTML file")
    parser.add_argument("--max-nodes", type=int, default=1000,
                        help="Maximum nodes to display (default: 100)")
    parser.add_argument("--filter-type", nargs="+", default=None,
                        help="Filter by entity types (e.g., ORGANIZATION PERSON)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only print statistics, don't generate visualization")
    
    args = parser.parse_args()
    
    print_stats(args.graphml)
    
    if not args.stats_only:
        visualize_kg(
            args.graphml,
            args.output,
            max_nodes=args.max_nodes,
            filter_by_type=args.filter_type
        )
