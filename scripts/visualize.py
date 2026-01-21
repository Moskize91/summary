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
    output_dir = project_root / "output" / "knowledge_graph"

    # Check if workspace exists
    if not workspace_path.exists():
        print(f"Error: Workspace not found at {workspace_path}")
        print("Please run the main extraction process first (python main.py)")
        return

    print(f"Loading workspace from: {workspace_path}")

    # Load Topologization object
    topo = Topologization(workspace_path)

    # Get all chapters
    chapter_ids = topo.get_all_chapter_ids()
    print(f"\nFound {len(chapter_ids)} chapter(s): {chapter_ids}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    total_visualizations = 0

    # Process each chapter and group
    for chapter_id in chapter_ids:
        group_ids = topo.get_group_ids_for_chapter(chapter_id)
        print(f"\nChapter {chapter_id}: {len(group_ids)} group(s)")

        for group_id in group_ids:
            print(f"\n{'=' * 60}")
            print(f"Processing Chapter {chapter_id}, Group {group_id}")
            print(f"{'=' * 60}")

            # Get snake graph for this chapter/group
            snake_graph = topo.get_snake_graph(chapter_id, group_id)

            # Collect snakes with metadata
            snakes = []
            snake_metadata = []
            for snake in snake_graph:
                snakes.append(snake.chunk_ids)
                snake_metadata.append(
                    {
                        "snake_id": snake.snake_id,
                        "tokens": snake.tokens,
                        "weight": snake.weight,
                        "size": snake.size,
                    }
                )

            if snakes:
                print(f"Found {len(snakes)} snake(s):")
                for i, snake_chunk_ids in enumerate(snakes):
                    chunk_first = topo.get_chunk(snake_chunk_ids[0])
                    chunk_last = topo.get_chunk(snake_chunk_ids[-1])
                    metadata = snake_metadata[i]
                    print(
                        f"  Snake {i}: {len(snake_chunk_ids)} nodes, {metadata['tokens']} tokens - "
                        f"{chunk_first.label} → {chunk_last.label}"
                    )
            else:
                print("No snakes detected (all nodes will be shown in gray)")

            # Build NetworkX graph for this group's chunks
            graph = nx.DiGraph()
            chunk_ids_in_group = set()

            # Collect all chunk IDs in this group
            for snake in snakes:
                chunk_ids_in_group.update(snake)

            # Add nodes with attributes
            for chunk_id in chunk_ids_in_group:
                chunk = topo.get_chunk(chunk_id)
                graph.add_node(
                    chunk.id,
                    generation=chunk.generation,
                    sentence_id=chunk.sentence_id,
                    label=chunk.label,
                    retention=chunk.retention,
                    importance=chunk.importance,
                )

            # Add edges (only edges within this group)
            for edge in topo.knowledge_graph.get_edges():
                if edge.from_chunk.id in chunk_ids_in_group and edge.to_chunk.id in chunk_ids_in_group:
                    graph.add_edge(edge.from_chunk.id, edge.to_chunk.id, strength=edge.strength)

            print(f"Group graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

            # Prepare graph_data for visualization
            graph_data = {
                "nodes": [],
                "edges": [
                    {"from": u, "to": v, "strength": data.get("strength")} for u, v, data in graph.edges(data=True)
                ],
            }

            # Load content for each node
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
                        "sentence_id": data.get("sentence_id", (0, 0, 0)),
                        "label": data.get("label", ""),
                        "retention": chunk.retention,
                        "importance": chunk.importance,
                        "metadata": metadata_str,
                        "content": chunk.content,
                    }
                )

            # Generate visualization with chapter-group prefix
            output_path = output_dir / f"chapter-{chapter_id}-group-{group_id}"
            print("\nGenerating visualization...")
            visualize_snakes(graph, snakes, output_path, graph_data, snake_metadata)
            total_visualizations += 1

    print(f"\n{'=' * 60}")
    print(f"Generated {total_visualizations} visualization(s) in {output_dir}")
    print(f"{'=' * 60}")

    # Close database connection
    topo.close()


if __name__ == "__main__":
    main()
