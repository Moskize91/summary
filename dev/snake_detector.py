"""Snake detector - re-exports from summary.snake_detector.

This module provides utility functions for loading and saving snake data,
and re-exports the main SnakeDetector class from the summary package.
"""

import json
from pathlib import Path

import networkx as nx

# Import the main implementation from summary package
from summary.snake_detector import SnakeDetector  # noqa: F401


def load_graph_from_json(json_path: Path) -> nx.DiGraph:
    """Load knowledge graph from JSON file.

    Args:
        json_path: Path to knowledge_graph.json

    Returns:
        NetworkX DiGraph
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    graph = nx.DiGraph()

    # Add nodes with attributes
    for node in data["nodes"]:
        graph.add_node(
            node["id"],
            sentence_id=node["sentence_id"],
            label=node["label"],
            content=node["content"],
        )

    # Add edges
    for edge in data["edges"]:
        graph.add_edge(edge["from"], edge["to"])

    return graph


def save_snakes_to_json(snakes: list[list[int]], output_path: Path, graph: nx.DiGraph) -> None:
    """Save detected snakes to JSON file.

    Args:
        snakes: List of snakes (each snake is a list of node IDs)
        output_path: Output JSON path
        graph: NetworkX graph for node metadata
    """
    snakes_data = []
    for i, snake in enumerate(snakes):
        snake_info = {
            "snake_id": i,
            "size": len(snake),
            "nodes": [
                {
                    "id": node_id,
                    "sentence_id": graph.nodes[node_id]["sentence_id"],
                    "label": graph.nodes[node_id]["label"],
                }
                for node_id in snake
            ],
        }
        snakes_data.append(snake_info)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snakes_data, f, ensure_ascii=False, indent=2)
