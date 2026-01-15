"""Snake detection using greedy edge merging with topological fingerprints.

Two-Phase Greedy Algorithm:

Phase 1: Snake Forming (Forbid Snake Eating Snake)
- Allow snakes to absorb isolated nodes
- **Forbidden**: Snake eating snake (both ≥2) - directly skip
- Run until no more valid merges (heap empty)

Phase 2: Snake Merging (Allow Snake Eating Snake)
- Remove the "forbid snake eating snake" restriction
- Allow all merges based on fingerprint similarity
- Stop when snake count ≤ total_nodes * 15%

This design first forms coherent snakes, then merges similar snakes into broader themes.
"""

import heapq
from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np


@dataclass
class MergeConfig:
    """Configuration for one merge phase."""

    max_cluster_size: int | None = None  # None = no limit
    reduction_ratio: float | None = None  # 0.25 = reduce by 25%, then stop
    stop_count: int | None = None  # Absolute number of clusters to stop at
    stop_ratio: float = 0.3  # Force stop when nodes <= initial * stop_ratio
    brake_ratio: float = 0.7  # Activate brake check when nodes <= initial * brake_ratio
    value_drop_threshold: float = 0.5  # Stop if value drops below prev * threshold
    enable_bonus: bool = False  # Enable bonus for large+singleton merges
    bonus_threshold: int = 3  # Minimum size for "large" cluster
    bonus_amount: float = 0.45  # Bonus multiplier (45% boost)


class SnakeDetector:
    """Detect thematic chains (snakes) using greedy edge merging algorithm.

    The algorithm:
    1. Compute topological fingerprints for each node (ink diffusion signatures)
    2. Initialize each node as a separate cluster
    3. Greedily merge clusters by selecting the highest-value edge
    4. Stop when node count reduces to target ratio or value drops significantly

    Edge value is defined as negative distance between clusters:
    - Initial: -distance(fingerprint_A, fingerprint_B)
    - After merge: -max_distance(all chunks in cluster_A, all chunks in cluster_B)
    """

    def __init__(
        self,
        max_hops: int = 100000,
        stop_ratio: float = 0.4,
        brake_ratio: float = 0.7,
        value_drop_threshold: float = 0.5,
        min_cluster_size: int = 3,
        distance_metric: Literal["max", "avg", "min"] = "max",
    ):
        """Initialize snake detector.

        Args:
            max_hops: Maximum hops for ink diffusion (1-5)
            stop_ratio: Force stop when node count <= initial * stop_ratio (default: 0.4)
            brake_ratio: Start checking value drop when node count <= initial * brake_ratio (default: 0.7)
            value_drop_threshold: Stop if current_value < prev_value * threshold (default: 0.5)
            min_cluster_size: Minimum nodes in a snake (default: 3)
            distance_metric: How to compute distance between clusters ("max", "avg", "min")
        """
        self.max_hops = max_hops
        self.stop_ratio = stop_ratio
        self.brake_ratio = brake_ratio
        self.value_drop_threshold = value_drop_threshold
        self.min_cluster_size = min_cluster_size
        self.distance_metric = distance_metric

    def detect_snakes(self, graph: nx.DiGraph) -> list[list[int]]:
        """Detect snakes in the knowledge graph using two-phase greedy merging.

        Args:
            graph: NetworkX directed graph with nodes having 'sentence_id', 'label', 'content'

        Returns:
            List of snakes, where each snake is a list of node IDs sorted by sentence_id
        """
        total_nodes = len(graph.nodes())

        # Step 1: Compute fingerprints for all nodes
        print("  Computing topological fingerprints...")
        fingerprints = self._compute_all_fingerprints(graph)

        # Phase 1: Snake Forming (forbid snake eating snake)
        print("\n  === Phase 1: Snake Forming (Forbid Snake Eating Snake) ===")
        phase1_config = MergeConfig(
            max_cluster_size=None,  # No limit
            reduction_ratio=None,  # No external stop
            stop_count=None,  # No absolute stop
            stop_ratio=0.0,  # Disabled - run until heap empty
            brake_ratio=0.0,  # Disabled
            value_drop_threshold=0.0,  # Disabled
            enable_bonus=True,  # Use to implement "forbid snake eating snake"
            bonus_threshold=2,  # >=2 is a snake
            bonus_amount=0.0,  # No actual bonus
        )
        clusters = self._greedy_merge(graph, fingerprints, phase1_config)

        # Phase 2: Snake Merging (allow snake eating snake)
        print("\n  === Phase 2: Snake Merging (Allow Snake Eating Snake) ===")
        phase2_stop_count = int(total_nodes * 0.15)
        print(f"  Target: Reduce to {phase2_stop_count} snakes ({0.15:.0%} of {total_nodes} total nodes)")

        phase2_config = MergeConfig(
            max_cluster_size=None,  # No limit
            reduction_ratio=None,  # No ratio-based stop
            stop_count=phase2_stop_count,  # Stop at 15% of total nodes
            stop_ratio=0.0,  # Disabled
            brake_ratio=0.0,  # Disabled
            value_drop_threshold=0.0,  # Disabled
            enable_bonus=False,  # Disable - allow snake eating snake
        )
        clusters = self._greedy_merge(graph, fingerprints, phase2_config, initial_clusters=clusters)

        # Step 3: Filter and sort
        snakes = []
        for cluster in clusters.values():
            # Sort by sentence_id
            cluster.sort(key=lambda nid: graph.nodes[nid]["sentence_id"])
            snakes.append(cluster)

        return snakes

    def _compute_all_fingerprints(self, graph: nx.DiGraph) -> dict[int, dict]:
        """Compute topological fingerprints for all nodes.

        Args:
            graph: NetworkX graph

        Returns:
            Dict mapping node_id to fingerprint dict (edge -> 1)
        """
        fingerprints = {}
        for node in graph.nodes():
            signature = self._compute_diffusion_signature(graph, node)
            fingerprints[node] = self._signature_to_fingerprint(signature)

        return fingerprints

    def _compute_diffusion_signature(self, graph: nx.DiGraph, node: int) -> dict:
        """Compute ink diffusion signature for a node.

        Args:
            graph: NetworkX directed graph
            node: Node ID

        Returns:
            Signature dict with 'outward' and 'inward' edge lists
        """
        signature = {"outward": {}, "inward": {}}

        # Outward diffusion (collect edges along out-direction)
        visited_out = {node}
        current_level_out = {node}
        for hop in range(1, self.max_hops + 1):
            next_level = set()
            edges = []
            for n in current_level_out:
                if n in graph:
                    for successor in graph.successors(n):
                        edges.append((n, successor))
                        if successor not in visited_out:
                            next_level.add(successor)
                            visited_out.add(successor)
            signature["outward"][hop] = edges
            current_level_out = next_level
            if not current_level_out:
                break

        # Inward diffusion (collect edges along in-direction)
        visited_in = {node}
        current_level_in = {node}
        for hop in range(1, self.max_hops + 1):
            next_level = set()
            edges = []
            for n in current_level_in:
                if n in graph:
                    for predecessor in graph.predecessors(n):
                        edges.append((predecessor, n))
                        if predecessor not in visited_in:
                            next_level.add(predecessor)
                            visited_in.add(predecessor)
            signature["inward"][hop] = edges
            current_level_in = next_level
            if not current_level_in:
                break

        return signature

    def _signature_to_fingerprint(self, signature: dict) -> dict:
        """Convert diffusion signature to edge-based fingerprint.

        Args:
            signature: Diffusion signature dict with edge lists

        Returns:
            Dict mapping edge tuples to presence (1)
        """
        fingerprint = {}

        # Collect all outward edges
        for hop_edges in signature["outward"].values():
            for edge in hop_edges:
                fingerprint[edge] = 1

        # Collect all inward edges
        for hop_edges in signature["inward"].values():
            for edge in hop_edges:
                fingerprint[edge] = 1

        return fingerprint

    def _compute_distance(self, fp1: dict, fp2: dict) -> float:
        """Compute normalized Euclidean distance between two edge-based fingerprints.

        Args:
            fp1: First fingerprint (dict of edges)
            fp2: Second fingerprint (dict of edges)

        Returns:
            Euclidean distance between normalized vectors
        """
        # Get union of all edges
        all_edges = sorted(set(fp1.keys()) | set(fp2.keys()))

        if not all_edges:
            return 0.0

        # Build sparse vectors
        vec1 = np.array([fp1.get(e, 0) for e in all_edges], dtype=float)
        vec2 = np.array([fp2.get(e, 0) for e in all_edges], dtype=float)

        # Normalize to unit vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 2.0  # Maximum distance for zero vectors

        vec1_normalized = vec1 / norm1
        vec2_normalized = vec2 / norm2

        # Euclidean distance between normalized vectors
        return float(np.linalg.norm(vec1_normalized - vec2_normalized))

    def _compute_cluster_distance(
        self, cluster1: list[int], cluster2: list[int], fingerprints: dict[int, dict]
    ) -> float:
        """Compute distance between two clusters.

        Args:
            cluster1: List of node IDs in first cluster
            cluster2: List of node IDs in second cluster
            fingerprints: Edge-based fingerprints

        Returns:
            Distance based on configured metric (max, avg, or min)
        """
        distances = []
        for node1 in cluster1:
            for node2 in cluster2:
                dist = self._compute_distance(fingerprints[node1], fingerprints[node2])
                distances.append(dist)

        if self.distance_metric == "max":
            return max(distances)
        elif self.distance_metric == "avg":
            return float(np.mean(distances))
        elif self.distance_metric == "min":
            return min(distances)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _greedy_merge(
        self,
        graph: nx.DiGraph,
        fingerprints: dict[int, dict],
        config: MergeConfig,
        initial_clusters: dict[int, list[int]] | None = None,
    ) -> dict[int, list[int]]:
        """Greedy edge merging algorithm with dynamic fingerprint updates.

        Args:
            graph: NetworkX graph
            fingerprints: Edge-based fingerprints for all nodes
            config: Configuration for this merge phase
            initial_clusters: Optional initial clusters (for phase 2)

        Returns:
            Dict mapping cluster_id to list of node IDs
        """
        # Create a dynamic graph for tracking merges
        # We'll remove internal edges as clusters merge
        dynamic_graph = graph.copy()

        # Initialize clusters
        if initial_clusters is None:
            # Phase 1: each node is a cluster
            clusters = {node_id: [node_id] for node_id in graph.nodes()}
            node_to_cluster = {node_id: node_id for node_id in graph.nodes()}
        else:
            # Phase 2: continue from phase 1 results
            clusters = {cid: list(nodes) for cid, nodes in initial_clusters.items()}
            node_to_cluster = {}
            for cluster_id, nodes in clusters.items():
                for node in nodes:
                    node_to_cluster[node] = cluster_id

            # Remove internal edges from dynamic graph
            for cluster_nodes in clusters.values():
                for n1 in cluster_nodes:
                    for n2 in cluster_nodes:
                        if n1 != n2 and dynamic_graph.has_edge(n1, n2):
                            dynamic_graph.remove_edge(n1, n2)

            # Recompute fingerprints with current dynamic graph
            for node_id in graph.nodes():
                signature = self._compute_diffusion_signature(dynamic_graph, node_id)
                fingerprints[node_id] = self._signature_to_fingerprint(signature)

        initial_count = len(clusters)

        # Compute stop conditions (priority: stop_count > reduction_ratio > stop_ratio)
        if config.stop_count is not None:
            # Use absolute stop count
            stop_count = config.stop_count
        elif config.reduction_ratio is not None:
            # Stop after reducing X% of nodes
            stop_count = int(initial_count * (1 - config.reduction_ratio))
        else:
            # Use relative stop ratio
            stop_count = int(initial_count * config.stop_ratio)

        brake_count = int(initial_count * config.brake_ratio)

        print(f"  Initial clusters: {initial_count}, stop at: {stop_count}, brake at: {brake_count}")

        # Build initial edge values
        edge_heap = []
        edge_set = set()  # Track which edges exist

        for cluster_u in clusters:
            for cluster_v in clusters:
                if cluster_u >= cluster_v:
                    continue
                if not self._clusters_connected(graph, clusters[cluster_u], clusters[cluster_v]):
                    continue

                dist = self._compute_cluster_distance(clusters[cluster_u], clusters[cluster_v], fingerprints)
                value = -dist  # Negative distance = higher value for smaller distance
                heapq.heappush(edge_heap, (-value, (cluster_u, cluster_v)))  # Max heap using negative value
                edge_set.add((cluster_u, cluster_v))

        prev_value = None
        brake_active = False

        while edge_heap and len(clusters) > stop_count:
            # Pop highest value edge
            neg_value, (u, v) = heapq.heappop(edge_heap)
            value = -neg_value

            # Check if edge still exists (nodes might have been merged)
            cluster_u = node_to_cluster.get(u)
            cluster_v = node_to_cluster.get(v)

            if cluster_u is None or cluster_v is None or cluster_u == cluster_v:
                continue  # Skip if nodes merged or don't exist

            # Check if edge still connects these clusters
            if not self._clusters_connected(graph, clusters[cluster_u], clusters[cluster_v]):
                continue

            # Check cluster size limit
            if config.max_cluster_size is not None:
                both_at_max = (
                    len(clusters[cluster_u]) >= config.max_cluster_size
                    and len(clusters[cluster_v]) >= config.max_cluster_size
                )
                if both_at_max:
                    continue  # Both clusters at max size, skip

            # Apply bonus if enabled (Phase 2)
            if config.enable_bonus:
                size_u = len(clusters[cluster_u])
                size_v = len(clusters[cluster_v])

                # Strict rule: Only allow snake (≥2) absorbing singleton (=1)
                # Forbidden: snake eating snake (both ≥2)
                both_snakes = size_u >= 2 and size_v >= 2
                if both_snakes:
                    continue  # Skip this merge entirely

                # Apply bonus for absorbing singleton
                absorb_singleton = (size_u >= 2 and size_v == 1) or (size_v >= 2 and size_u == 1)
                if absorb_singleton:
                    # Add bonus: value = value + abs(value) * bonus_amount
                    value = value + abs(value) * config.bonus_amount
                    # Update neg_value for consistency
                    neg_value = -value

            # Activate brake check when reaching brake_count
            if len(clusters) <= brake_count:
                brake_active = True

            # Brake check: stop if value drops significantly
            if brake_active and prev_value is not None:
                if value < prev_value * config.value_drop_threshold:
                    print(f"  Brake activated: value dropped from {prev_value:.4f} to {value:.4f}")
                    break

            # === MERGE OPERATION WITH GLOBAL FINGERPRINT UPDATE ===

            # Step 1: Merge cluster_v into cluster_u
            clusters[cluster_u].extend(clusters[cluster_v])
            for node in clusters[cluster_v]:
                node_to_cluster[node] = cluster_u
            del clusters[cluster_v]

            # Step 2: Remove internal edges from dynamic graph
            merged_nodes = clusters[cluster_u]
            edges_to_remove = set()  # Use set to avoid duplicates
            for n1 in merged_nodes:
                for n2 in merged_nodes:
                    if n1 != n2:
                        if dynamic_graph.has_edge(n1, n2):
                            edges_to_remove.add((n1, n2))
                        if dynamic_graph.has_edge(n2, n1):
                            edges_to_remove.add((n2, n1))

            for edge in edges_to_remove:
                if dynamic_graph.has_edge(*edge):  # Double-check before removal
                    dynamic_graph.remove_edge(*edge)

            # Step 3: Recompute fingerprints for ALL nodes (global update)
            # This ensures all nodes see the removal of internal edges
            for node_id in graph.nodes():
                signature = self._compute_diffusion_signature(dynamic_graph, node_id)
                fingerprints[node_id] = self._signature_to_fingerprint(signature)

            # Step 4: Update edges involving merged cluster
            neighbors = set()
            for node in merged_nodes:
                if node in graph:
                    neighbors.update(graph.successors(node))
                    neighbors.update(graph.predecessors(node))

            # Recompute edge values for all neighboring clusters
            for neighbor in neighbors:
                neighbor_cluster = node_to_cluster.get(neighbor)
                if neighbor_cluster is None or neighbor_cluster == cluster_u:
                    continue

                # Compute new distance between merged cluster and neighbor cluster
                new_dist = self._compute_cluster_distance(clusters[cluster_u], clusters[neighbor_cluster], fingerprints)
                new_value = -new_dist

                # Add to heap (old edges will be filtered out by the check above)
                heapq.heappush(edge_heap, (-new_value, (cluster_u, neighbor_cluster)))

            prev_value = value

            # Progress reporting
            if len(clusters) % 5 == 0:
                ratio = len(clusters) / initial_count
                print(f"    Clusters: {len(clusters)} ({ratio:.1%}), Edge value: {value:.4f}")

        print(f"  Final clusters: {len(clusters)} ({len(clusters) / initial_count:.1%} of initial)")
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
