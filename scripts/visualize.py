"""Script to generate visualization from knowledge graph JSON with snake detection."""

import json
from pathlib import Path

from dev.snake_detector import SnakeDetector, load_graph_from_json, save_snakes_to_json
from dev.visualize_snakes import visualize_snakes


def main():
    """Generate visualization with snake detection from the latest knowledge graph output."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    json_path = project_root / "output" / "knowledge_graph.json"
    output_path = project_root / "output" / "knowledge_graph"

    # Check if JSON exists
    if not json_path.exists():
        print(f"Error: Knowledge graph JSON not found at {json_path}")
        print("Please run the main extraction process first.")
        return

    print(f"Reading knowledge graph from: {json_path}")

    # Load graph
    graph = load_graph_from_json(json_path)
    print(f"Loaded graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")

    # Load original graph data for visualization
    with open(json_path, encoding="utf-8") as f:
        graph_data = json.load(f)

    # Detect snakes using greedy merging algorithm
    print("\nDetecting thematic chains (snakes)...")
    detector = SnakeDetector(
        max_hops=3,  # Ink diffusion range
        stop_ratio=0.3,  # Force stop at 30% of initial nodes
        brake_ratio=0.7,  # Start checking value drop at 70%
        value_drop_threshold=0.5,  # Stop if value drops below 50% of previous
        min_cluster_size=3,  # Minimum snake length (filter out pairs)
        distance_metric="max",  # Use max distance between clusters
    )
    snakes = detector.detect_snakes(graph)

    if snakes:
        print(f"Found {len(snakes)} snakes:")
        for i, snake in enumerate(snakes):
            first_node = graph.nodes[snake[0]]
            last_node = graph.nodes[snake[-1]]
            print(f"  Snake {i}: {len(snake)} nodes - {first_node['label']} → {last_node['label']}")

        # Save snakes to JSON
        snakes_output = project_root / "output" / "snakes.json"
        save_snakes_to_json(snakes, snakes_output, graph)
    else:
        print("No snakes detected (all nodes will be shown in gray)")

    # Generate visualization
    print("\nGenerating visualization...")
    visualize_snakes(graph, snakes, output_path, graph_data)


if __name__ == "__main__":
    main()
