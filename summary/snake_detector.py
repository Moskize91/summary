"""Snake detection using greedy edge merging with topological fingerprints."""

import heapq
from typing import Literal

import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler


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
        max_hops: int = 3,
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
        """Detect snakes in the knowledge graph.

        Args:
            graph: NetworkX directed graph with nodes having 'sentence_id', 'label', 'content'

        Returns:
            List of snakes, where each snake is a list of node IDs sorted by sentence_id
        """
        if len(graph.nodes()) < self.min_cluster_size:
            return []

        # Step 1: Compute fingerprints for all nodes
        print("  Computing topological fingerprints...")
        fingerprints = self._compute_all_fingerprints(graph)

        # Step 2: Greedy merging
        print("  Greedy edge merging...")
        clusters = self._greedy_merge(graph, fingerprints)

        # Step 3: Filter and sort
        snakes = []
        for cluster in clusters.values():
            if len(cluster) >= self.min_cluster_size:
                # Sort by sentence_id
                cluster.sort(key=lambda nid: graph.nodes[nid]["sentence_id"])
                snakes.append(cluster)

        return snakes

    def _compute_all_fingerprints(self, graph: nx.DiGraph) -> dict[int, np.ndarray]:
        """Compute topological fingerprints for all nodes.

        Args:
            graph: NetworkX graph

        Returns:
            Dict mapping node_id to fingerprint vector
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
            Signature dict with 'outward' and 'inward' diffusion patterns
        """
        signature = {"outward": {}, "inward": {}}

        # Outward diffusion (along out-edges, toward future)
        visited_out = {node}
        current_level_out = {node}
        for hop in range(1, self.max_hops + 1):
            next_level = set()
            for n in current_level_out:
                if n in graph:
                    for successor in graph.successors(n):
                        if successor not in visited_out:
                            next_level.add(successor)
                            visited_out.add(successor)
            signature["outward"][hop] = sorted(next_level)
            current_level_out = next_level
            if not current_level_out:
                break

        # Inward diffusion (along in-edges, toward past)
        visited_in = {node}
        current_level_in = {node}
        for hop in range(1, self.max_hops + 1):
            next_level = set()
            for n in current_level_in:
                if n in graph:
                    for predecessor in graph.predecessors(n):
                        if predecessor not in visited_in:
                            next_level.add(predecessor)
                            visited_in.add(predecessor)
            signature["inward"][hop] = sorted(next_level)
            current_level_in = next_level
            if not current_level_in:
                break

        return signature

    def _signature_to_fingerprint(self, signature: dict) -> np.ndarray:
        """Convert diffusion signature to numerical fingerprint vector.

        Args:
            signature: Diffusion signature dict

        Returns:
            NumPy array of features
        """
        features = []

        # Feature 1-3: Number of outward neighbors at each hop
        for hop in range(1, self.max_hops + 1):
            count = len(signature["outward"].get(hop, []))
            features.append(count)

        # Feature 4-6: Number of inward neighbors at each hop
        for hop in range(1, self.max_hops + 1):
            count = len(signature["inward"].get(hop, []))
            features.append(count)

        # Feature 7: Total diffusion range
        all_neighbors = set()
        for hop_neighbors in signature["outward"].values():
            all_neighbors.update(hop_neighbors)
        for hop_neighbors in signature["inward"].values():
            all_neighbors.update(hop_neighbors)
        features.append(len(all_neighbors))

        # Feature 8: Directionality (outward / inward ratio)
        total_out = sum(len(neighbors) for neighbors in signature["outward"].values())
        total_in = sum(len(neighbors) for neighbors in signature["inward"].values())
        directionality = total_out / (total_in + 1e-6)
        features.append(directionality)

        # Feature 9: Max outward depth
        max_out_hop = max(signature["outward"].keys()) if signature["outward"] else 0
        features.append(max_out_hop)

        # Feature 10: Max inward depth
        max_in_hop = max(signature["inward"].keys()) if signature["inward"] else 0
        features.append(max_in_hop)

        return np.array(features, dtype=float)

    def _normalize_fingerprints(self, fingerprints: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """Normalize all fingerprints using StandardScaler.

        Args:
            fingerprints: Dict mapping node_id to raw fingerprint

        Returns:
            Dict mapping node_id to normalized fingerprint
        """
        node_ids = sorted(fingerprints.keys())
        vectors = np.vstack([fingerprints[nid] for nid in node_ids])

        scaler = StandardScaler()
        vectors_normalized = scaler.fit_transform(vectors)

        return {nid: vectors_normalized[i] for i, nid in enumerate(node_ids)}

    def _compute_distance(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Compute cosine distance between two fingerprints.

        Args:
            fp1: First fingerprint
            fp2: Second fingerprint

        Returns:
            Cosine distance (1 - cosine_similarity)
        """
        # Cosine similarity: dot(a, b) / (norm(a) * norm(b))
        norm1 = np.linalg.norm(fp1)
        norm2 = np.linalg.norm(fp2)

        # Handle zero vectors
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance for zero vectors

        cosine_sim = np.dot(fp1, fp2) / (norm1 * norm2)
        # Cosine distance: 1 - cosine_similarity (range: [0, 2])
        return float(1 - cosine_sim)

    def _compute_cluster_distance(
        self, cluster1: list[int], cluster2: list[int], fingerprints: dict[int, np.ndarray]
    ) -> float:
        """Compute distance between two clusters.

        Args:
            cluster1: List of node IDs in first cluster
            cluster2: List of node IDs in second cluster
            fingerprints: Normalized fingerprints

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

    def _greedy_merge(self, graph: nx.DiGraph, fingerprints: dict[int, np.ndarray]) -> dict[int, list[int]]:
        """Greedy edge merging algorithm.

        Args:
            graph: NetworkX graph
            fingerprints: Raw fingerprints for all nodes

        Returns:
            Dict mapping cluster_id to list of node IDs
        """
        # Normalize fingerprints
        fingerprints = self._normalize_fingerprints(fingerprints)

        # Initialize: each node is a cluster
        clusters = {node_id: [node_id] for node_id in graph.nodes()}
        node_to_cluster = {node_id: node_id for node_id in graph.nodes()}

        initial_count = len(clusters)
        stop_count = int(initial_count * self.stop_ratio)
        brake_count = int(initial_count * self.brake_ratio)

        # Build initial edge values
        edge_heap = []
        edge_set = set()  # Track which edges exist

        for u, v in graph.edges():
            dist = self._compute_distance(fingerprints[u], fingerprints[v])
            value = -dist  # Negative distance = higher value for smaller distance
            heapq.heappush(edge_heap, (-value, (u, v)))  # Max heap using negative value
            edge_set.add((u, v))

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

            # Activate brake check when reaching brake_count
            if len(clusters) <= brake_count:
                brake_active = True

            # Brake check: stop if value drops significantly
            if brake_active and prev_value is not None:
                if value < prev_value * self.value_drop_threshold:
                    print(f"  Brake activated: value dropped from {prev_value:.4f} to {value:.4f}")
                    break

            # Merge: cluster_v into cluster_u
            clusters[cluster_u].extend(clusters[cluster_v])
            for node in clusters[cluster_v]:
                node_to_cluster[node] = cluster_u
            del clusters[cluster_v]

            # Update edges involving merged cluster
            # Find all neighbors of the merged cluster
            neighbors = set()
            for node in clusters[cluster_u]:
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
