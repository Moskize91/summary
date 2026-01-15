"""Script to generate visualization from knowledge graph JSON with snake detection."""

import json
from pathlib import Path

from dev.snake_detector import SnakeDetector, load_graph_from_json, save_snakes_to_json
from dev.visualize_snakes import visualize_snakes
from summary.snake_detector import split_connected_components


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

    # Split into connected components
    print("\nSplitting into connected components...")
    components = split_connected_components(graph)
    print(f"Found {len(components)} connected component(s):")
    for i, comp in enumerate(components):
        print(f"  Component {i}: {len(comp.nodes())} nodes, {len(comp.edges())} edges")

    # Detect snakes in each component
    print("\nDetecting thematic chains (snakes)...")
    detector = SnakeDetector(
        min_cluster_size=2,  # Minimum snake length (include pairs)
        phase2_stop_ratio=0.15,  # Phase 2 stops at 15% of component nodes
    )

    all_snakes = []
    for i, component in enumerate(components):
        print(f"\nProcessing Component {i}:")
        component_snakes = detector.detect_snakes(component)
        all_snakes.extend(component_snakes)

    snakes = all_snakes

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
