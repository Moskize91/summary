"""Script to generate snake-level visualization from summaries."""

import json
from pathlib import Path

from dev.snake_detector import load_graph_from_json
from dev.visualize_snake_graph import visualize_snake_graph
from summary.topologization.snake_graph_builder import SnakeGraphBuilder


def main():
    """Generate snake-level visualization from snake summaries."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    json_path = project_root / "output" / "knowledge_graph.json"
    snakes_path = project_root / "output" / "snakes.json"
    summaries_path = project_root / "output" / "snake_summaries.json"
    output_path = project_root / "output" / "snake_graph"

    # Check if files exist
    if not json_path.exists():
        print(f"Error: Knowledge graph JSON not found at {json_path}")
        print("Please run the main extraction process first.")
        return

    if not snakes_path.exists():
        print(f"Error: Snakes JSON not found at {snakes_path}")
        print("Please run the main extraction process with snake detection first.")
        return

    if not summaries_path.exists():
        print(f"Error: Snake summaries JSON not found at {summaries_path}")
        print("Please run the main extraction process with summarization first.")
        return

    print(f"Reading knowledge graph from: {json_path}")
    graph = load_graph_from_json(json_path)
    print(f"Loaded graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")

    print(f"\nReading snakes from: {snakes_path}")
    with open(snakes_path, encoding="utf-8") as f:
        snakes_data = json.load(f)

    # Extract snake node lists
    snakes = [snake_data["nodes"] for snake_data in snakes_data]
    # Each snake_data["nodes"] is a list of dicts with "id" key
    # Convert to list of node IDs
    snakes = [[node["id"] for node in snake_nodes] for snake_nodes in snakes]
    print(f"Loaded {len(snakes)} snakes")

    print(f"\nReading snake summaries from: {summaries_path}")
    with open(summaries_path, encoding="utf-8") as f:
        snake_summaries = json.load(f)
    print(f"Loaded {len(snake_summaries)} summaries")

    # Build snake-level graph
    print("\nBuilding snake-level graph...")
    builder = SnakeGraphBuilder()
    snake_graph = builder.build_snake_graph(snakes, graph)

    print(f"Snake graph: {len(snake_graph.nodes())} snakes, {len(snake_graph.edges())} inter-snake edges")

    # Generate visualization
    print("\nGenerating visualization...")
    visualize_snake_graph(snake_graph, output_path, snake_summaries)


if __name__ == "__main__":
    main()
