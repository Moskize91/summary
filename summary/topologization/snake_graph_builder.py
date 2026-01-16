"""Build snake-level graph from detected snakes and knowledge graph."""

import networkx as nx


class SnakeGraphBuilder:
    """Build a higher-level graph where nodes are snakes and edges connect related snakes.

    Filters out intra-snake edges (edges between nodes within the same snake)
    and creates inter-snake edges by checking connections between nodes in different snakes.
    """

    def build_snake_graph(
        self,
        snakes: list[list[int]],
        knowledge_graph: nx.DiGraph,
    ) -> nx.DiGraph:
        """Build snake-level graph from knowledge graph and detected snakes.

        Args:
            snakes: List of snakes (each is a list of node IDs)
            knowledge_graph: Original graph with node-level connections

        Returns:
            Snake-level directed graph with:
                - Nodes: snake IDs (0, 1, 2, ...)
                - Node attributes: size, first_label, last_label, node_ids
                - Edges: (snake_from, snake_to)
                - Edge attributes: internal_edge_count
        """
        # Build node_to_snake mapping
        node_to_snake = {}
        for snake_id, snake in enumerate(snakes):
            for node_id in snake:
                node_to_snake[node_id] = snake_id

        # Create snake graph
        snake_graph = nx.DiGraph()

        # Add snake nodes with attributes
        for snake_id, snake in enumerate(snakes):
            first_node = knowledge_graph.nodes[snake[0]]
            last_node = knowledge_graph.nodes[snake[-1]]

            snake_graph.add_node(
                snake_id,
                size=len(snake),
                first_label=first_node["label"],
                last_label=last_node["label"],
                node_ids=snake,
            )

        # Find inter-snake edges
        # Group edges by (snake_from, snake_to) pair
        inter_snake_edges = {}  # (snake_from, snake_to) -> count

        for edge in knowledge_graph.edges():
            node_from, node_to = edge

            # Check if both nodes belong to snakes
            if node_from not in node_to_snake or node_to not in node_to_snake:
                continue

            snake_from = node_to_snake[node_from]
            snake_to = node_to_snake[node_to]

            # Filter out intra-snake edges
            if snake_from == snake_to:
                continue

            # Add to inter-snake edge count
            snake_edge_key = (snake_from, snake_to)
            if snake_edge_key not in inter_snake_edges:
                inter_snake_edges[snake_edge_key] = 0

            inter_snake_edges[snake_edge_key] += 1

        # Create snake-level edges
        # Normalize edge direction based on snake's starting sentence_id
        for (snake_from, snake_to), edge_count in inter_snake_edges.items():
            # Get starting sentence_id for both snakes
            snake_from_start_sid = knowledge_graph.nodes[snakes[snake_from][0]]["sentence_id"]
            snake_to_start_sid = knowledge_graph.nodes[snakes[snake_to][0]]["sentence_id"]

            # Normalize direction: always from earlier snake to later snake
            if snake_from_start_sid < snake_to_start_sid:
                # snake_from is earlier, keep direction
                final_from, final_to = snake_from, snake_to
            else:
                # snake_to is earlier, reverse direction
                final_from, final_to = snake_to, snake_from

            # Add edge to snake graph (or update if already exists)
            if snake_graph.has_edge(final_from, final_to):
                # Edge already exists, accumulate count
                existing_count = snake_graph.edges[final_from, final_to]["internal_edge_count"]
                snake_graph.edges[final_from, final_to]["internal_edge_count"] = existing_count + edge_count
            else:
                # Add new edge
                snake_graph.add_edge(
                    final_from,
                    final_to,
                    internal_edge_count=edge_count,
                )

        return snake_graph
