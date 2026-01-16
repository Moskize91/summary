"""Calculate importance scores for edges in a knowledge graph.

This module provides methods to evaluate edge importance from multiple perspectives:
1. Structural importance (bridges, betweenness)
2. Information flow importance (how many paths go through)
3. Local clustering importance (contribution to local structure)
4. Node importance (importance of connected nodes)
"""

import networkx as nx


class EdgeImportanceCalculator:
    """Calculate importance scores for edges in a knowledge graph.

    Provides multiple metrics to evaluate edge importance:
    - Edge betweenness: measures information flow
    - Bridge detection: identifies critical structural edges
    - Node importance product: edges connecting important nodes
    - Clustering impact: contribution to local clustering structure
    """

    def __init__(self, graph: nx.DiGraph):
        """Initialize calculator with a directed graph.

        Args:
            graph: Input directed graph (can be disconnected)
        """
        self.graph = graph
        # Use undirected for structural analysis
        self.undirected = graph.to_undirected()

    def compute_edge_betweenness(self) -> dict[frozenset, float]:
        """Compute edge betweenness centrality.

        Edge betweenness measures how many shortest paths pass through an edge.
        High betweenness = edge is on many shortest paths = important for information flow.

        Returns:
            Dict mapping edge (frozenset of two nodes) to betweenness score
        """
        # NetworkX returns dict with tuple keys
        betweenness = nx.edge_betweenness_centrality(self.undirected)

        # Convert to frozenset keys for consistency
        return {frozenset([u, v]): score for (u, v), score in betweenness.items()}

    def compute_bridge_scores(self) -> dict[frozenset, float]:
        """Compute bridge scores for edges.

        A bridge is an edge whose removal increases the number of connected components.
        Bridges are critical for maintaining graph connectivity.

        Returns:
            Dict mapping edge to 1.0 (bridge) or 0.0 (non-bridge)
        """
        bridges = set(nx.bridges(self.undirected))

        scores = {}
        for u, v in self.undirected.edges():
            edge = frozenset([u, v])
            # Check if edge is a bridge (in either direction)
            is_bridge = (u, v) in bridges or (v, u) in bridges
            scores[edge] = 1.0 if is_bridge else 0.0

        return scores

    def compute_node_importance_product(self) -> dict[frozenset, float]:
        """Compute edge importance based on connected node importance.

        An edge connecting two important nodes is itself important.
        Node importance is measured by degree (number of connections).

        Returns:
            Dict mapping edge to product of endpoint degrees
        """
        scores = {}
        for u, v in self.undirected.edges():
            edge = frozenset([u, v])
            # Use degree as node importance metric
            degree_u = self.undirected.degree(u)
            degree_v = self.undirected.degree(v)
            scores[edge] = degree_u * degree_v

        return scores

    def compute_clustering_impact(self) -> dict[frozenset, float]:
        """Compute edge importance based on local clustering structure.

        Edges that connect nodes with many common neighbors contribute to
        local clustering structure. More common neighbors = stronger local connection.

        Returns:
            Dict mapping edge to number of common neighbors
        """
        scores = {}
        for u, v in self.undirected.edges():
            edge = frozenset([u, v])
            # Count common neighbors
            neighbors_u = set(self.undirected.neighbors(u))
            neighbors_v = set(self.undirected.neighbors(v))
            common_neighbors = len(neighbors_u & neighbors_v)
            scores[edge] = float(common_neighbors)

        return scores

    def compute_combined_importance(
        self,
        weights: dict[str, float] | None = None,
        normalize: bool = True,
    ) -> dict[frozenset, float]:
        """Compute combined importance score from multiple metrics.

        Args:
            weights: Dict of metric weights. Default:
                - betweenness: 0.4 (information flow)
                - bridge: 0.3 (structural criticality)
                - node_product: 0.2 (node importance)
                - clustering: 0.1 (local structure)
            normalize: Whether to normalize final scores to [0, 1]

        Returns:
            Dict mapping edge to combined importance score
        """
        # Default weights emphasize information flow and structural importance
        if weights is None:
            weights = {
                "betweenness": 0.4,
                "bridge": 0.3,
                "node_product": 0.2,
                "clustering": 0.1,
            }

        print("  Computing edge betweenness centrality...")
        betweenness = self.compute_edge_betweenness()

        print("  Detecting bridge edges...")
        bridge = self.compute_bridge_scores()

        print("  Computing node importance products...")
        node_product = self.compute_node_importance_product()

        print("  Computing clustering impact...")
        clustering = self.compute_clustering_impact()

        # Normalize each metric to [0, 1] for fair combination
        print("  Normalizing metrics...")
        betweenness_norm = self._normalize_scores(betweenness)
        node_product_norm = self._normalize_scores(node_product)
        clustering_norm = self._normalize_scores(clustering)
        # Bridge scores already 0 or 1

        # Combine with weights
        combined = {}
        for edge in betweenness_norm.keys():
            score = (
                weights["betweenness"] * betweenness_norm[edge]
                + weights["bridge"] * bridge[edge]
                + weights["node_product"] * node_product_norm[edge]
                + weights["clustering"] * clustering_norm[edge]
            )
            combined[edge] = score

        # Final normalization if requested
        if normalize:
            combined = self._normalize_scores(combined)

        return combined

    def _normalize_scores(self, scores: dict[frozenset, float]) -> dict[frozenset, float]:
        """Normalize scores to [0, 1] range.

        Args:
            scores: Dict of scores to normalize

        Returns:
            Dict with normalized scores
        """
        if not scores:
            return scores

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)

        # Handle case where all values are the same
        if max_val == min_val:
            return {k: 0.5 for k in scores}

        # Linear normalization
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

    def get_top_edges(
        self,
        scores: dict[frozenset, float],
        k: int = 10,
    ) -> list[tuple[frozenset, float]]:
        """Get top K most important edges.

        Args:
            scores: Edge importance scores
            k: Number of top edges to return

        Returns:
            List of (edge, score) tuples sorted by score (descending)
        """
        sorted_edges = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_edges[:k]

    def get_bottom_edges(
        self,
        scores: dict[frozenset, float],
        k: int = 10,
    ) -> list[tuple[frozenset, float]]:
        """Get bottom K least important edges.

        Args:
            scores: Edge importance scores
            k: Number of bottom edges to return

        Returns:
            List of (edge, score) tuples sorted by score (ascending)
        """
        sorted_edges = sorted(scores.items(), key=lambda x: x[1])
        return sorted_edges[:k]
