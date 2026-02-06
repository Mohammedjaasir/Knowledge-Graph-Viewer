#!/usr/bin/env python3
"""
Export Knowledge Graph to JSON

Converts the GraphML knowledge graph to JSON format for web visualization.

Usage:
    python export_graph.py
"""

import networkx as nx
import json
from pathlib import Path


def export_graph_to_json(graphml_path="rag_storage/graph_chunk_entity_relation.graphml",
                         output_path="graph_data.json"):
    """Export GraphML to JSON format for vis.js"""

    print("Loading knowledge graph...")
    G = nx.read_graphml(graphml_path)

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    # Convert nodes
    nodes = []
    entity_types = {}

    for node_id, node_data in G.nodes(data=True):
        entity_type = node_data.get('entity_type', 'unknown')

        # Count entity types
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        # Get first sentence of description for label
        description = node_data.get('description', '')
        first_sentence = description.split('<SEP>')[0] if description else node_id
        if len(first_sentence) > 100:
            first_sentence = first_sentence[:100] + "..."

        nodes.append({
            'id': node_id,
            'label': node_id,
            'title': first_sentence,  # Tooltip
            'group': entity_type,
            'entity_type': entity_type,
            'description': description,
            'source_id': node_data.get('source_id', ''),
            'file_path': node_data.get('file_path', '')
        })

    # Convert edges
    edges = []
    for source, target, edge_data in G.edges(data=True):
        weight = edge_data.get('weight', 1.0)
        description = edge_data.get('description', '')
        keywords = edge_data.get('keywords', '')

        # Get first sentence for tooltip
        first_desc = description.split('<SEP>')[0] if description else ''
        if len(first_desc) > 150:
            first_desc = first_desc[:150] + "..."

        edges.append({
            'from': source,
            'to': target,
            'value': float(weight),  # Edge thickness
            'title': first_desc,  # Tooltip
            'weight': float(weight),
            'description': description,
            'keywords': keywords
        })

    # Export to JSON
    graph_data = {
        'nodes': nodes,
        'edges': edges,
        'stats': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'entity_types': entity_types
        }
    }

    print(f"\nExporting to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Exported successfully!")
    print(f"\nEntity Type Distribution:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count}")

    return output_path


if __name__ == "__main__":
    export_graph_to_json()
