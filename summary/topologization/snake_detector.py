"""Snake detection using greedy edge merging with topological fingerprints.

Two-Phase Algorithm:

Phase 1: Snake Forming (Forbid Snake Eating Snake)
- Allow snakes to absorb isolated nodes
- Forbidden: Snake eating snake (both ≥2)
- Token limit: merged tokens must not exceed snake_tokens
- Run until no more valid merges

Phase 2: Snake Merging (Allow Snake Eating Snake)
- Allow all merges based on fingerprint similarity
- Token limit: merged tokens must not exceed snake_tokens
- Stop when no more valid merges (all remaining merges exceed token limit)

Design Principles:
1. Input MUST be a weakly connected graph (use split_connected_components first)
2. All nodes have fingerprints with IDENTICAL structure (same nodes) but DIFFERENT values
3. Fingerprint values are normalized ink concentrations from node to all other nodes
4. Distance metric is the diameter of merged cluster (maximum pairwise distance)
5. Diameter-based distance creates natural self-correction without artificial limits
6. Edge weights based on node attributes (retention/importance) and link strength
7. Ink flows from source node through weighted edges to all nodes (BFS, treating edges as undirected)
8. Token-based snake size control: each snake's total tokens cannot exceed snake_tokens
"""

import heapq
from dataclasses import dataclass

import networkx as nx

from .enums import ImportanceLevel, LinkStrength, RetentionLevel


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

    snake_tokens: int  # Maximum tokens allowed in a snake
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
        snake_tokens: int = 700,
    ):
        """Initialize snake detector.

        Args:
            min_cluster_size: Minimum nodes in a snake (default: 2)
            snake_tokens: Maximum tokens allowed in a snake (default: 700)
        """
        self.min_cluster_size = min_cluster_size
        self.snake_tokens = snake_tokens

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

        # Check total tokens - if small enough, return as single snake
        total_tokens = sum(graph.nodes[node].get("tokens", 0) for node in graph.nodes())
        print(f"  Total nodes: {total_nodes}, Total tokens: {total_tokens}")

        if total_tokens <= self.snake_tokens:
            print(f"  Total tokens ({total_tokens}) ≤ snake_tokens ({self.snake_tokens})")
            print(f"  Returning entire component as single snake")
            snake = list(graph.nodes())
            snake.sort(key=lambda nid: graph.nodes[nid]["sentence_id"])
            return [snake]

        # Step 1: Compute fingerprints for all nodes
        print("  Computing topological fingerprints...")
        fingerprints = self._compute_all_fingerprints(graph)

        # Validate all fingerprints are identical
        self._validate_fingerprints(fingerprints)

        # Phase 1: Snake Forming (forbid snake eating snake)
        print("\n  === Phase 1: Snake Forming (Forbid Snake Eating Snake) ===")
        print(f"  Initial clusters: {total_nodes}")

        phase1_config = MergeConfig(
            snake_tokens=self.snake_tokens,
            enable_bonus=True,  # Enable to forbid snake eating snake
        )
        clusters = self._greedy_merge(graph, fingerprints, phase1_config)

        # Phase 2: Snake Merging (allow snake eating snake)
        print("\n  === Phase 2: Snake Merging (Allow Snake Eating Snake) ===")
        print(f"  Target: Merge snakes until no more valid merges (token limit: {self.snake_tokens})")

        phase2_config = MergeConfig(
            snake_tokens=self.snake_tokens,
            enable_bonus=False,  # Disable - allow snake eating snake
        )
        clusters = self._greedy_merge(graph, fingerprints, phase2_config, initial_clusters=clusters)

        # Step 3: Filter and sort (include all clusters, even single nodes)
        snakes = []
        for cluster in clusters.values():
            cluster.sort(key=lambda nid: graph.nodes[nid]["sentence_id"])
            snakes.append(cluster)

        return snakes

    def _compute_node_weights(self, graph: nx.DiGraph) -> dict[int, float]:
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

    def _compute_edge_weights(self, graph: nx.DiGraph) -> dict[tuple, float]:
        """Compute weight for each edge based on node weights and link strength.

        Edge weight calculation:
        1. Compute half-weight for each endpoint node:
           - Get all edges (in+out) connected to node
           - Compute link strength ratio for current edge
           - Half-weight = node_weight * link_strength_ratio
        2. Edge weight = sum of two half-weights
        3. Ensure minimum weight of 0.1 to allow ink diffusion even without attributes

        Args:
            graph: NetworkX graph with node and edge attributes

        Returns:
            Dict mapping (u, v) edge tuple to weight (undirected, both orders)
        """
        node_weights = self._compute_node_weights(graph)
        edge_weights = {}

        # Minimum edge weight to ensure diffusion even without node attributes
        MIN_EDGE_WEIGHT = 0.1

        # Process each edge (treat as undirected for weight computation)
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

            # Ensure minimum weight for ink diffusion
            final_weight = max(edge_weight, MIN_EDGE_WEIGHT)

            # Store for both directions (undirected for weight purposes)
            edge_weights[(u, v)] = final_weight
            edge_weights[(v, u)] = final_weight

        return edge_weights

    def _compute_all_fingerprints(self, graph: nx.DiGraph) -> dict[int, dict]:
        """Compute fingerprints for all nodes using weighted ink diffusion with decay.

        Each fingerprint is a dict mapping nodes to normalized ink concentrations.
        - Ink starts from source node (concentration 1.0)
        - Flows through weighted edges to all other nodes (treating edges as undirected)
        - Each generation (BFS layer) applies decay factor: concentration × 0.65
        - Each node can only be colored once (BFS layer-by-layer)
        - Nodes in same layer can receive ink from multiple sources (cumulative)
        - After all nodes are colored, normalize concentrations to sum=1.0

        Args:
            graph: NetworkX graph with node/edge attributes

        Returns:
            Dict mapping node_id to fingerprint dict {target_node_id: normalized_concentration}
        """
        # Decay factor applied per generation (BFS layer)
        DECAY_FACTOR = 0.75

        edge_weights = self._compute_edge_weights(graph)

        fingerprints = {}
        for start_node in graph.nodes():
            # BFS with weighted ink diffusion
            concentrations = {start_node: 1.0}
            current_layer = [start_node]
            visited = {start_node}

            while current_layer:
                next_layer_nodes = {}  # {node: cumulative_concentration}

                for current_node in current_layer:
                    current_concentration = concentrations[current_node]

                    # Find all unvisited neighbors (treat graph as undirected)
                    neighbors = []
                    total_weight = 0.0

                    # Check in_edges (predecessors)
                    for predecessor, _ in graph.in_edges(current_node):
                        if predecessor not in visited:
                            # Edge weight is symmetric (undirected)
                            weight = edge_weights.get(
                                (predecessor, current_node), edge_weights.get((current_node, predecessor), 1.0)
                            )
                            neighbors.append((predecessor, weight))
                            total_weight += weight

                    # Check out_edges (successors)
                    for _, successor in graph.out_edges(current_node):
                        if successor not in visited:
                            weight = edge_weights.get(
                                (current_node, successor), edge_weights.get((successor, current_node), 1.0)
                            )
                            neighbors.append((successor, weight))
                            total_weight += weight

                    # Distribute concentration proportionally by edge weights with decay
                    if total_weight > 0:
                        for neighbor, weight in neighbors:
                            contribution = current_concentration * (weight / total_weight) * DECAY_FACTOR
                            if neighbor not in next_layer_nodes:
                                next_layer_nodes[neighbor] = 0.0
                            next_layer_nodes[neighbor] += contribution

                # Add next layer nodes to concentrations and visited set
                for node, concentration in next_layer_nodes.items():
                    concentrations[node] = concentration
                    visited.add(node)

                current_layer = list(next_layer_nodes.keys())

            # Normalize concentrations (sum to 1.0)
            total = sum(concentrations.values())
            if total > 0:
                for node in concentrations:
                    concentrations[node] /= total

            fingerprints[start_node] = concentrations

        return fingerprints

    def _validate_fingerprints(self, fingerprints: dict[int, dict]) -> None:
        """Validate that all fingerprints have the same structure (same nodes).

        Args:
            fingerprints: Dict of node fingerprints

        Raises:
            AssertionError: If fingerprints don't have the same node set
        """
        if not fingerprints:
            return

        # Get reference node set
        first_node = next(iter(fingerprints))
        reference_nodes = set(fingerprints[first_node].keys())

        # Check all others have the same nodes (but can have different concentrations)
        for node, fp in fingerprints.items():
            fp_nodes = set(fp.keys())
            if fp_nodes != reference_nodes:
                raise AssertionError(
                    f"Fingerprint structure mismatch! Node {node} has different nodes. "
                    f"Expected {len(reference_nodes)} nodes, got {len(fp_nodes)} nodes. "
                    "This should never happen in a connected graph."
                )

        print(f"  ✓ Validated: All {len(fingerprints)} nodes have identical structure ({len(reference_nodes)} nodes)")

    def _compute_merge_value(
        self,
        graph: nx.DiGraph,
        cluster1: list[int],
        cluster2: list[int],
        fingerprints: dict[int, dict],
    ) -> float:
        """Compute merge value for two clusters based on their connecting edges.

        Value = minimum edge value among all connecting edges between clusters.
        Edge value = (1 - d/2.0) * dv
        where:
        - d: Euclidean distance between node fingerprints (range: 0.0 to 2.0 for unit vectors)
        - dv: link_strength + (3 if both nodes have retention) + (3 if both have importance)

        Taking minimum ensures that the weakest connection determines merge priority.
        Higher values indicate better merge candidates (closer distance, stronger links).

        Args:
            graph: NetworkX graph with node/edge attributes
            cluster1: First cluster nodes
            cluster2: Second cluster nodes
            fingerprints: Node fingerprints (dict mapping node_id to concentration dict)

        Returns:
            Merge value (higher is better, range 0.0 to 15.0)
        """
        min_value = float("inf")
        edge_count = 0

        # Enumerate all edges between the two clusters
        for u in cluster1:
            for v in cluster2:
                # Check if edge exists (either direction)
                edge_data = None
                if graph.has_edge(u, v):
                    edge_data = graph.edges[u, v]
                elif graph.has_edge(v, u):
                    edge_data = graph.edges[v, u]

                if edge_data is None:
                    continue

                edge_count += 1

                # Compute distance d between fingerprints (Euclidean distance)
                squared_sum = 0.0
                for target_node in fingerprints[u]:  # All nodes have same structure
                    diff = fingerprints[u][target_node] - fingerprints[v][target_node]
                    squared_sum += diff * diff
                d = squared_sum**0.5

                # Compute dv (bonus value from link strength and node attributes)
                strength_str = edge_data.get("strength")
                link_strength = LinkStrength.from_string(strength_str)
                dv = float(link_strength.value) if link_strength else 1.0

                # Add bonuses for node attributes
                u_data = graph.nodes[u]
                v_data = graph.nodes[v]

                if u_data.get("retention") and v_data.get("retention"):
                    dv += 3.0
                if u_data.get("importance") and v_data.get("importance"):
                    dv += 3.0

                # Compute edge value: v = (1 - d/2.0) * dv
                base_value = 1.0 - d / 2.0
                value = base_value * dv

                min_value = min(min_value, value)

        # If no connecting edges found, return very low value
        if edge_count == 0:
            return -float("inf")

        return min_value

    def _compute_cluster_tokens(self, graph: nx.DiGraph, cluster: list[int]) -> int:
        """Compute total tokens for a cluster.

        Args:
            graph: NetworkX graph with node attributes
            cluster: List of node IDs

        Returns:
            Total token count
        """
        return sum(graph.nodes[node].get("tokens", 0) for node in cluster)

    def _greedy_merge(
        self,
        graph: nx.DiGraph,
        fingerprints: dict[int, dict],
        config: MergeConfig,
        initial_clusters: dict[int, list[int]] | None = None,
    ) -> dict[int, list[int]]:
        """Greedy merge algorithm with token limit.

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

        print(f"  Initial clusters: {initial_count}, token limit: {config.snake_tokens}")

        # Build initial edge heap
        edge_heap = []
        edge_set = set()

        for cluster_u in clusters:
            for cluster_v in clusters:
                if cluster_u >= cluster_v:
                    continue
                if not self._clusters_connected(graph, clusters[cluster_u], clusters[cluster_v]):
                    continue

                merge_value = self._compute_merge_value(graph, clusters[cluster_u], clusters[cluster_v], fingerprints)
                value = merge_value

                heapq.heappush(edge_heap, (-value, (cluster_u, cluster_v)))
                edge_set.add((cluster_u, cluster_v))

        # Greedy merging loop (continue until heap is empty)
        while edge_heap:
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

            # Token limit check: ensure merged cluster doesn't exceed token limit
            tokens_u = self._compute_cluster_tokens(graph, clusters[cluster_u])
            tokens_v = self._compute_cluster_tokens(graph, clusters[cluster_v])
            merged_tokens = tokens_u + tokens_v

            if merged_tokens > config.snake_tokens:
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

                # Compute new merge value
                new_merge_value = self._compute_merge_value(
                    graph, clusters[cluster_u], clusters[neighbor_cluster], fingerprints
                )
                new_value = new_merge_value

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
