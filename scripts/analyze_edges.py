"""Script to analyze edge importance in knowledge graph."""

from pathlib import Path

from dev.snake_detector import load_graph_from_json
from summary.edge_importance import EdgeImportanceCalculator


def main():
    """Analyze edge importance in the knowledge graph."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    json_path = project_root / "output" / "knowledge_graph.json"

    # Check if JSON exists
    if not json_path.exists():
        print(f"Error: Knowledge graph JSON not found at {json_path}")
        print("Please run the main extraction process first.")
        return

    print(f"Reading knowledge graph from: {json_path}")

    # Load graph
    graph = load_graph_from_json(json_path)
    print(f"Loaded graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges\n")

    # Initialize calculator
    calculator = EdgeImportanceCalculator(graph)

    # Compute combined importance
    print("Computing edge importance scores...")
    importance_scores = calculator.compute_combined_importance()

    # Show statistics
    print(f"\nEdge importance statistics:")
    scores = list(importance_scores.values())
    print(f"  Min score: {min(scores):.4f}")
    print(f"  Max score: {max(scores):.4f}")
    print(f"  Mean score: {sum(scores) / len(scores):.4f}")

    # Show top important edges
    print(f"\n{'=' * 80}")
    print("Top 10 Most Important Edges:")
    print(f"{'=' * 80}")
    top_edges = calculator.get_top_edges(importance_scores, k=10)
    for i, (edge, score) in enumerate(top_edges, 1):
        u, v = sorted(edge)  # Get nodes from frozenset
        u_label = graph.nodes[u]["label"]
        v_label = graph.nodes[v]["label"]
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   {u_label} ↔ {v_label}")

    # Show bottom unimportant edges
    print(f"\n{'=' * 80}")
    print("Bottom 10 Least Important Edges:")
    print(f"{'=' * 80}")
    bottom_edges = calculator.get_bottom_edges(importance_scores, k=10)
    for i, (edge, score) in enumerate(bottom_edges, 1):
        u, v = sorted(edge)
        u_label = graph.nodes[u]["label"]
        v_label = graph.nodes[v]["label"]
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   {u_label} ↔ {v_label}")

    # Analyze individual metrics
    print(f"\n{'=' * 80}")
    print("Individual Metric Analysis:")
    print(f"{'=' * 80}")

    print("\nBridge edges (critical for connectivity):")
    bridge_scores = calculator.compute_bridge_scores()
    bridges = [edge for edge, score in bridge_scores.items() if score == 1.0]
    print(f"  Found {len(bridges)} bridge edges out of {len(bridge_scores)} total edges")
    if bridges:
        print("  Examples:")
        for edge in bridges[:5]:
            u, v = sorted(edge)
            u_label = graph.nodes[u]["label"]
            v_label = graph.nodes[v]["label"]
            print(f"    - {u_label} ↔ {v_label}")

    print("\nTop 5 edges by betweenness (information flow):")
    betweenness = calculator.compute_edge_betweenness()
    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
    for edge, score in top_betweenness:
        u, v = sorted(edge)
        u_label = graph.nodes[u]["label"]
        v_label = graph.nodes[v]["label"]
        print(f"    Score: {score:.4f} - {u_label} ↔ {v_label}")


if __name__ == "__main__":
    main()
