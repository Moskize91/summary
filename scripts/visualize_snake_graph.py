"""Script to generate snake-level visualization from summaries."""

from pathlib import Path

import networkx as nx

from dev.visualize_snake_graph import visualize_snake_graph
from summary.topologization import Topologization


def main():
    """Generate snake-level visualization from workspace."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    workspace_path = project_root / "workspace"
    output_path = project_root / "output" / "snake_graph"

    # Check if workspace exists
    if not workspace_path.exists():
        print(f"Error: Workspace not found at {workspace_path}")
        print("Please run the main extraction process first (python main.py)")
        return

    print(f"Loading workspace from: {workspace_path}")

    # Load Topologization object
    topo = Topologization(workspace_path)

    # Build NetworkX graph from snake graph
    snake_graph = nx.DiGraph()

    # Add nodes with attributes and collect summaries
    snake_summaries = []
    for snake in topo.snake_graph:
        snake_graph.add_node(
            snake.snake_id,
            size=snake.size,
            first_label=snake.first_label,
            last_label=snake.last_label,
            node_ids=snake.chunk_ids,  # Add chunk IDs to node attributes
        )

        # Collect summary for visualization
        summary_dict = {
            "snake_id": snake.snake_id,
            "first_label": snake.first_label,
            "last_label": snake.last_label,
            "summary": snake.summary,
            "size": snake.size,
        }
        snake_summaries.append(summary_dict)

    # Add edges with internal edge counts
    for from_id, to_id, count in topo.snake_graph.get_edges():
        snake_graph.add_edge(from_id, to_id, internal_edge_count=count)

    print(f"Loaded snake graph with {len(snake_graph.nodes())} snakes and {len(snake_graph.edges())} inter-snake edges")

    # Calculate importance for edges (normalize internal_edge_count)
    if snake_graph.number_of_edges() > 0:
        max_count = max(snake_graph.edges[edge]["internal_edge_count"] for edge in snake_graph.edges())
        for edge in snake_graph.edges():
            count = snake_graph.edges[edge]["internal_edge_count"]
            # Normalize to [0, 1]
            importance = count / max_count if max_count > 0 else 0.5
            snake_graph.edges[edge]["importance"] = importance

    print(f"Loaded {len(snake_summaries)} summaries")

    # Generate visualization
    print("\nGenerating visualization...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_snake_graph(snake_graph, output_path, snake_summaries)

    # Close database connection
    topo.close()


if __name__ == "__main__":
    main()
