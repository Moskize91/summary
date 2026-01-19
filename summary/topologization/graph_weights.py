"""Utilities for computing node and edge weights in knowledge graph."""

import networkx as nx

from .enums import ImportanceLevel, LinkStrength, RetentionLevel


def compute_node_weights(graph: nx.DiGraph) -> dict[int, float]:
    """Compute weight for each node based on retention and importance.

    Node weight = retention_value + importance_value
    - Both attributes are optional, use value if present

    Args:
        graph: NetworkX graph with node attributes

    Returns:
        Dict mapping node_id to weight
    """
    node_weights = {}
    for node in graph.nodes():
        node_data = graph.nodes[node]
        weight = 0.0

        # Add retention value if present
        retention_str = node_data.get("retention")
        if retention_str:
            retention = RetentionLevel.from_string(retention_str)
            if retention:
                weight += float(retention.value)

        # Add importance value if present
        importance_str = node_data.get("importance")
        if importance_str:
            importance = ImportanceLevel.from_string(importance_str)
            if importance:
                weight += float(importance.value)

        node_weights[node] = weight

    return node_weights


def compute_edge_weights(graph: nx.DiGraph) -> dict[tuple, float]:
    """Compute weight for each edge based on node weights and link strength.

    Edge weight calculation:
    1. Compute half-weight for each endpoint node:
       - Get all edges (in+out) connected to node
       - Compute link strength ratio for current edge
       - Half-weight = node_weight * link_strength_ratio
    2. Edge weight = sum of two half-weights
    3. Ensure minimum weight of 0.1 to allow connections even without attributes

    Args:
        graph: NetworkX graph with node and edge attributes

    Returns:
        Dict mapping (u, v) edge tuple to weight (directed)
    """
    node_weights = compute_node_weights(graph)
    edge_weights = {}

    # Minimum edge weight to ensure connections even without node attributes
    MIN_EDGE_WEIGHT = 0.1

    # Process each edge
    for u, v in graph.edges():
        edge_data = graph.edges[u, v]
        strength_str = edge_data.get("strength")
        link_strength = LinkStrength.from_string(strength_str)
        edge_link_strength = float(link_strength.value) if link_strength else 1.0

        # Compute half-weight for node u
        u_edges = list(graph.in_edges(u)) + list(graph.out_edges(u))
        u_total_strength = 0.0
        for edge in u_edges:
            edge_data_u = graph.edges[edge]
            strength_u = LinkStrength.from_string(edge_data_u.get("strength"))
            u_total_strength += float(strength_u.value) if strength_u else 1.0

        u_half_weight = node_weights[u] * (edge_link_strength / u_total_strength if u_total_strength > 0 else 0)

        # Compute half-weight for node v
        v_edges = list(graph.in_edges(v)) + list(graph.out_edges(v))
        v_total_strength = 0.0
        for edge in v_edges:
            edge_data_v = graph.edges[edge]
            strength_v = LinkStrength.from_string(edge_data_v.get("strength"))
            v_total_strength += float(strength_v.value) if strength_v else 1.0

        v_half_weight = node_weights[v] * (edge_link_strength / v_total_strength if v_total_strength > 0 else 0)

        # Edge weight = sum of two half-weights
        edge_weight = u_half_weight + v_half_weight

        # Ensure minimum weight
        final_weight = max(edge_weight, MIN_EDGE_WEIGHT)

        # Store for this direction
        edge_weights[(u, v)] = final_weight

    return edge_weights


def add_weights_to_graph(graph: nx.DiGraph) -> None:
    """Add weight attributes to graph nodes and edges in-place.

    Adds:
    - node attribute 'weight': node weight based on retention/importance
    - edge attribute 'weight': edge weight based on nodes and link strength

    Args:
        graph: NetworkX graph to modify
    """
    # Compute and add node weights
    node_weights = compute_node_weights(graph)
    for node_id, weight in node_weights.items():
        graph.nodes[node_id]["weight"] = weight

    # Compute and add edge weights
    edge_weights = compute_edge_weights(graph)
    for (u, v), weight in edge_weights.items():
        graph.edges[u, v]["weight"] = weight
