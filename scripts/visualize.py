"""Script to generate visualization from workspace with snake detection."""

from pathlib import Path

import networkx as nx

from dev.visualize_snakes import visualize_snakes
from summary.topologization import Topologization


def main():
    """Generate visualization with snake detection from workspace."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    workspace_path = project_root / "workspace"
    output_path = project_root / "output" / "knowledge_graph"

    # Check if workspace exists
    if not workspace_path.exists():
        print(f"Error: Workspace not found at {workspace_path}")
        print("Please run the main extraction process first (python main.py)")
        return

    print(f"Loading workspace from: {workspace_path}")

    # Load Topologization object
    topo = Topologization(workspace_path)

    # Build NetworkX graph from knowledge graph
    graph = nx.DiGraph()

    # Add nodes with attributes
    for chunk in topo.knowledge_graph:
        graph.add_node(
            chunk.id,
            generation=chunk.generation,
            sentence_id=chunk.sentence_id,
            label=chunk.label,
            retention=chunk.retention,
            importance=chunk.importance,
            # Don't add content to save memory
        )

    # Add edges
    for edge in topo.knowledge_graph.get_edges():
        graph.add_edge(edge.from_chunk.id, edge.to_chunk.id, strength=edge.strength)

    print(f"Loaded graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")

    # Collect snakes from snake graph with metadata
    snakes = []
    snake_metadata = []
    for snake in topo.snake_graph:
        snakes.append(snake.chunk_ids)
        snake_metadata.append({
            "snake_id": snake.snake_id,
            "tokens": snake.tokens,
            "weight": snake.weight,
            "size": snake.size,
        })

    if snakes:
        print(f"\nFound {len(snakes)} snakes:")
        for i, snake_chunk_ids in enumerate(snakes):
            first_node = graph.nodes[snake_chunk_ids[0]]
            last_node = graph.nodes[snake_chunk_ids[-1]]
            metadata = snake_metadata[i]
            print(
                f"  Snake {i}: {len(snake_chunk_ids)} nodes, {metadata['tokens']} tokens - "
                f"{first_node['label']} → {last_node['label']}"
            )
    else:
        print("\nNo snakes detected (all nodes will be shown in gray)")

    # Prepare graph_data for visualization (convert node attributes to dict format)
    # Note: We need to manually load content for visualization
    graph_data = {
        "nodes": [],
        "edges": [{"from": u, "to": v, "strength": data.get("strength")} for u, v, data in graph.edges(data=True)],
    }

    # Load content from Topologization for each node
    for node_id, data in graph.nodes(data=True):
        chunk = topo.get_chunk(node_id)

        # Format retention/importance for display
        attrs = []
        if chunk.retention:
            attrs.append(f"retention:{chunk.retention}")
        if chunk.importance:
            attrs.append(f"importance:{chunk.importance}")
        metadata_str = ", ".join(attrs) if attrs else ""

        graph_data["nodes"].append(
            {
                "id": node_id,
                "generation": data.get("generation", 0),
                "sentence_id": data.get("sentence_id", (0, 0)),
                "label": data.get("label", ""),
                "retention": chunk.retention,
                "importance": chunk.importance,
                "metadata": metadata_str,  # For easy display
                "content": chunk.content,  # AI-generated summary from database
            }
        )

    # Generate visualization
    print("\nGenerating visualization...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_snakes(graph, snakes, output_path, graph_data, snake_metadata)

    # Close database connection
    topo.close()


if __name__ == "__main__":
    main()
