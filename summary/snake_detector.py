"""Snake detection using greedy edge merging with topological fingerprints.

Two-Phase Algorithm:

Phase 1: Snake Forming (Forbid Snake Eating Snake)
- Allow snakes to absorb isolated nodes
- Forbidden: Snake eating snake (both ≥2)
- Run until no more valid merges

Phase 2: Snake Merging (Allow Snake Eating Snake)
- Allow all merges based on fingerprint similarity
- Stop when snake count ≤ total_nodes * phase2_stop_ratio

Design Principles:
1. Input MUST be a weakly connected graph (use split_connected_components first)
2. All nodes have fingerprints with IDENTICAL structure (same edges) but DIFFERENT values
3. Fingerprint values are distances from node to each edge (ink diffusion without decay)
4. Distance metric is the diameter of merged cluster (maximum pairwise distance)
5. Diameter-based distance creates natural self-correction without artificial limits
"""

import heapq
from dataclasses import dataclass

import networkx as nx


def split_connected_components(graph: nx.DiGraph) -> list[nx.DiGraph]:
    """Split a directed graph into weakly connected components.

    Args:
        graph: Input directed graph

    Returns:
        List of subgraphs, each being a weakly connected component
    """
    components = list(nx.weakly_connected_components(graph))
    subgraphs = []

    for component in components:
        subgraph = graph.subgraph(component).copy()
        subgraphs.append(subgraph)

    return subgraphs


@dataclass
class MergeConfig:
    """Configuration for one merge phase."""

    stop_count: int | None = None  # Absolute number of clusters to stop at
    enable_bonus: bool = False  # Enable bonus for snake+singleton merges (forbid snake eating snake)


class SnakeDetector:
    """Detect thematic chains (snakes) using greedy edge merging.

    Requirements:
    - Input graph MUST be weakly connected (use split_connected_components first)
    - All nodes will have identical fingerprints (all edges in graph)
    - Distance is based on internal vs external edges in clusters
    """

    def __init__(
        self,
        min_cluster_size: int = 2,
        phase2_stop_ratio: float = 0.15,
    ):
        """Initialize snake detector.

        Args:
            min_cluster_size: Minimum nodes in a snake (default: 2)
            phase2_stop_ratio: Phase 2 stops at this ratio of total nodes (default: 0.15)
        """
        self.min_cluster_size = min_cluster_size
        self.phase2_stop_ratio = phase2_stop_ratio

    def detect_snakes(self, graph: nx.DiGraph) -> list[list[int]]:
        """Detect snakes in a weakly connected graph.

        Args:
            graph: NetworkX directed graph (MUST be weakly connected)

        Returns:
            List of snakes, where each snake is a list of node IDs sorted by sentence_id

        Raises:
            ValueError: If graph is not weakly connected
        """
        # Validate input is connected
        if not nx.is_weakly_connected(graph):
            raise ValueError(
                "Input graph must be weakly connected. Use split_connected_components() to split the graph first."
            )

        total_nodes = len(graph.nodes())

        # Step 1: Compute fingerprints for all nodes
        print("  Computing topological fingerprints...")
        fingerprints = self._compute_all_fingerprints(graph)

        # Validate all fingerprints are identical
        self._validate_fingerprints(fingerprints)

        # Phase 1: Snake Forming (forbid snake eating snake)
        print("\n  === Phase 1: Snake Forming (Forbid Snake Eating Snake) ===")
        print(f"  Initial clusters: {total_nodes}")

        phase1_config = MergeConfig(
            stop_count=0,  # Run until no more valid merges
            enable_bonus=True,  # Enable to forbid snake eating snake
            # No max_snake_size limit - rely on diameter self-correction
        )
        clusters = self._greedy_merge(graph, fingerprints, phase1_config)

        # Phase 2: Snake Merging (allow snake eating snake)
        print("\n  === Phase 2: Snake Merging (Allow Snake Eating Snake) ===")
        phase2_stop_count = int(total_nodes * self.phase2_stop_ratio)
        print(
            f"  Target: Reduce to {phase2_stop_count} snakes "
            f"({self.phase2_stop_ratio:.0%} of {total_nodes} total nodes)"
        )

        phase2_config = MergeConfig(
            stop_count=phase2_stop_count,
            enable_bonus=False,  # Disable - allow snake eating snake
            # No size penalty - rely on diameter self-correction
        )
        clusters = self._greedy_merge(graph, fingerprints, phase2_config, initial_clusters=clusters)

        # Step 3: Filter and sort (include all clusters, even single nodes)
        snakes = []
        for cluster in clusters.values():
            cluster.sort(key=lambda nid: graph.nodes[nid]["sentence_id"])
            snakes.append(cluster)

        return snakes

    def _compute_all_fingerprints(self, graph: nx.DiGraph) -> dict[int, dict]:
        """Compute fingerprints for all nodes.

        Each fingerprint is a dict mapping edges to distances from the node.
        - All fingerprints contain the same edges (structure identical)
        - But distance values differ based on node position in graph
        - Ink diffusion without decay: distance = shortest path to edge

        Args:
            graph: NetworkX graph

        Returns:
            Dict mapping node_id to fingerprint dict {edge: distance}
        """
        # Convert to undirected for distance calculation
        undirected = graph.to_undirected()

        fingerprints = {}
        for node in graph.nodes():
            # BFS from this node to get shortest path distances
            distances = nx.single_source_shortest_path_length(undirected, node)

            # Build fingerprint: {edge: distance_to_edge}
            fp = {}
            for u, v in graph.edges():
                edge = frozenset([u, v])
                # Distance to edge = min distance to either endpoint
                dist_to_edge = min(
                    distances.get(u, float("inf")),
                    distances.get(v, float("inf")),
                )
                fp[edge] = dist_to_edge

            fingerprints[node] = fp

        return fingerprints

    def _validate_fingerprints(self, fingerprints: dict[int, dict]) -> None:
        """Validate that all fingerprints have the same structure (same edges).

        Args:
            fingerprints: Dict of node fingerprints

        Raises:
            AssertionError: If fingerprints don't have the same edge set
        """
        if not fingerprints:
            return

        # Get reference edge set
        first_node = next(iter(fingerprints))
        reference_edges = set(fingerprints[first_node].keys())

        # Check all others have the same edges (but can have different distances)
        for node, fp in fingerprints.items():
            node_edges = set(fp.keys())
            if node_edges != reference_edges:
                raise AssertionError(
                    f"Fingerprint structure mismatch! Node {node} has different edges. "
                    f"Expected {len(reference_edges)} edges, got {len(node_edges)} edges. "
                    "This should never happen in a connected graph."
                )

        print(f"  ✓ Validated: All {len(fingerprints)} nodes have identical structure ({len(reference_edges)} edges)")

    def _compute_distance(
        self, cluster1: list[int], cluster2: list[int], graph: nx.DiGraph, fingerprints: dict[int, dict]
    ) -> float:
        """Compute distance as the diameter of merged cluster.

        Distance = maximum pairwise distance within the merged cluster.
        This creates self-correction: loose clusters have large diameter and won't easily merge.

        Args:
            cluster1: First cluster nodes
            cluster2: Second cluster nodes
            graph: Original graph (unused, kept for compatibility)
            fingerprints: Node fingerprints

        Returns:
            Maximum distance between any two nodes in merged cluster (diameter)
        """
        # Simulate merge
        merged = cluster1 + cluster2

        # Find maximum pairwise distance (diameter)
        max_distance = 0.0
        for i, node_i in enumerate(merged):
            for node_j in merged[i + 1 :]:  # Avoid duplicate pairs
                # Euclidean distance between fingerprints
                squared_sum = 0.0
                for edge in fingerprints[node_i]:  # All nodes have same edges
                    diff = fingerprints[node_i][edge] - fingerprints[node_j][edge]
                    squared_sum += diff * diff

                distance = squared_sum**0.5
                max_distance = max(max_distance, distance)

        return max_distance

    def _greedy_merge(
        self,
        graph: nx.DiGraph,
        fingerprints: dict[int, dict],
        config: MergeConfig,
        initial_clusters: dict[int, list[int]] | None = None,
    ) -> dict[int, list[int]]:
        """Greedy merge algorithm.

        Args:
            graph: NetworkX graph
            fingerprints: Fingerprints for all nodes (dict mapping edges to distances)
            config: Configuration for this merge phase
            initial_clusters: Optional initial clusters (for phase 2)

        Returns:
            Dict mapping cluster_id to list of node IDs
        """
        # Initialize clusters
        if initial_clusters is None:
            # Phase 1: each node is a cluster
            clusters = {node_id: [node_id] for node_id in graph.nodes()}
            node_to_cluster = {node_id: node_id for node_id in graph.nodes()}
        else:
            # Phase 2: continue from phase 1
            clusters = {cid: list(nodes) for cid, nodes in initial_clusters.items()}
            node_to_cluster = {}
            for cluster_id, nodes in clusters.items():
                for node in nodes:
                    node_to_cluster[node] = cluster_id

        initial_count = len(clusters)
        stop_count = config.stop_count if config.stop_count is not None else 0

        print(f"  Initial clusters: {initial_count}, stop at: {stop_count}")

        # Build initial edge heap
        edge_heap = []
        edge_set = set()

        for cluster_u in clusters:
            for cluster_v in clusters:
                if cluster_u >= cluster_v:
                    continue
                if not self._clusters_connected(graph, clusters[cluster_u], clusters[cluster_v]):
                    continue

                dist = self._compute_distance(clusters[cluster_u], clusters[cluster_v], graph, fingerprints)
                value = -dist  # Negative distance for max heap

                heapq.heappush(edge_heap, (-value, (cluster_u, cluster_v)))
                edge_set.add((cluster_u, cluster_v))

        # Greedy merging loop
        while edge_heap and len(clusters) > stop_count:
            # Pop highest value edge
            neg_value, (u, v) = heapq.heappop(edge_heap)
            value = -neg_value

            # Check if edge still valid
            cluster_u = node_to_cluster.get(u)
            cluster_v = node_to_cluster.get(v)

            if cluster_u is None or cluster_v is None or cluster_u == cluster_v:
                continue

            if not self._clusters_connected(graph, clusters[cluster_u], clusters[cluster_v]):
                continue

            # Phase 1: Forbid snake eating snake
            if config.enable_bonus:
                size_u = len(clusters[cluster_u])
                size_v = len(clusters[cluster_v])
                both_snakes = size_u >= 2 and size_v >= 2

                if both_snakes:
                    continue  # Skip this merge

            # Merge operation
            clusters[cluster_u].extend(clusters[cluster_v])
            for node in clusters[cluster_v]:
                node_to_cluster[node] = cluster_u
            del clusters[cluster_v]

            # Update neighboring edges
            merged_nodes = clusters[cluster_u]
            neighbors = set()
            for node in merged_nodes:
                if node in graph:
                    neighbors.update(graph.successors(node))
                    neighbors.update(graph.predecessors(node))

            # Recompute edges to neighbor clusters
            for neighbor in neighbors:
                neighbor_cluster = node_to_cluster.get(neighbor)
                if neighbor_cluster is None or neighbor_cluster == cluster_u:
                    continue

                # Compute new distance
                new_dist = self._compute_distance(clusters[cluster_u], clusters[neighbor_cluster], graph, fingerprints)
                new_value = -new_dist

                # Add to heap
                if cluster_u < neighbor_cluster:
                    edge_key = (cluster_u, neighbor_cluster)
                else:
                    edge_key = (neighbor_cluster, cluster_u)

                if edge_key not in edge_set:
                    heapq.heappush(edge_heap, (-new_value, edge_key))
                    edge_set.add(edge_key)

            # Progress reporting
            if len(clusters) % 5 == 0 or len(clusters) <= 20:
                progress = len(clusters) / initial_count * 100
                print(f"    Clusters: {len(clusters)} ({progress:.1f}%), Edge value: {value:.4f}")

        print(f"  Final clusters: {len(clusters)} ({len(clusters) / initial_count * 100:.1f}% of initial)")

        return clusters

    def _clusters_connected(self, graph: nx.DiGraph, cluster1: list[int], cluster2: list[int]) -> bool:
        """Check if two clusters have at least one edge connecting them.

        Args:
            graph: NetworkX graph
            cluster1: First cluster
            cluster2: Second cluster

        Returns:
            True if clusters are connected by at least one edge
        """
        for u in cluster1:
            if u not in graph:
                continue
            for v in cluster2:
                if graph.has_edge(u, v) or graph.has_edge(v, u):
                    return True
        return False
