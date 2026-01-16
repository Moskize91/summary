"""Wave Reflection algorithm for working memory selection and importance scoring."""

from collections import deque

import networkx as nx

from .cognitive_chunk import CognitiveChunk


class WaveReflection:
    """Implements the Wave Reflection algorithm for chunk importance scoring.

    The algorithm consists of two phases:
    1. Forward propagation: from latest chunks, propagate scores backward along edges
    2. Reflection: reverse propagation with generation decay

    Edge direction: always from later chunks to earlier chunks (based on sentence_id).
    """

    def __init__(self, generation_decay_factor: float = 0.9):
        """Initialize Wave Reflection algorithm.

        Args:
            generation_decay_factor: Decay factor for older generations (0-1)
        """
        self.generation_decay_factor = generation_decay_factor

    def select_top_chunks(
        self,
        all_chunks: list[CognitiveChunk],
        knowledge_graph: nx.DiGraph,
        latest_chunk_ids: list[int],
        capacity: int,
    ) -> list[CognitiveChunk]:
        """Select top M chunks using Wave Reflection algorithm.

        Args:
            all_chunks: All available chunks (including old chunks)
            knowledge_graph: Knowledge graph with edges
            latest_chunk_ids: IDs of chunks from the latest extraction
            capacity: Number of chunks to select (M)

        Returns:
            Top M chunks sorted by importance
        """
        if not all_chunks:
            return []

        if not latest_chunk_ids:
            # No new chunks, just return existing chunks (no scoring change)
            return all_chunks[:capacity]

        # Build chunk ID to chunk mapping
        chunk_map = {chunk.id: chunk for chunk in all_chunks}

        # Lock latest chunks (they must be selected)
        locked_chunks = [chunk_map[cid] for cid in latest_chunk_ids if cid in chunk_map]

        # If locked chunks already fill capacity, return them
        if len(locked_chunks) >= capacity:
            return locked_chunks[:capacity]

        # Calculate remaining slots
        remaining_capacity = capacity - len(locked_chunks)
        locked_chunk_ids = set(latest_chunk_ids)

        # Phase 1: Forward propagation
        forward_scores = self._forward_propagation(
            latest_chunk_ids=latest_chunk_ids,
            knowledge_graph=knowledge_graph,
        )

        if not forward_scores:
            # No reachable nodes, return only locked chunks
            return locked_chunks

        # Phase 2: Reflection propagation
        reflection_scores = self._reflection_propagation(forward_scores=forward_scores, knowledge_graph=knowledge_graph)

        # Score only non-locked chunks
        candidate_scores = []
        for chunk_id, reflection_score in reflection_scores.items():
            # Skip locked chunks
            if chunk_id in locked_chunk_ids:
                continue

            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue

            # Apply generation decay to reflection score
            generation = chunk.generation
            decayed_score = reflection_score * (self.generation_decay_factor**generation)

            candidate_scores.append((chunk, decayed_score))

        # Sort by score (descending) and select top remaining_capacity
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        selected_from_candidates = [chunk for chunk, _ in candidate_scores[:remaining_capacity]]

        # Combine locked chunks with selected candidates
        return locked_chunks + selected_from_candidates

    def _forward_propagation(
        self,
        latest_chunk_ids: list[int],
        knowledge_graph: nx.DiGraph,
    ) -> dict[int, float]:
        """Phase 1: Forward propagation along edges (from later to earlier).

        Args:
            latest_chunk_ids: Starting chunk IDs (latest extraction)
            knowledge_graph: Knowledge graph with directed edges

        Returns:
            Dictionary mapping chunk ID to forward propagation score
        """
        # Initialize scores for latest chunks
        N = len(latest_chunk_ids)
        initial_score = 1.0 / N

        scores = {}
        for chunk_id in latest_chunk_ids:
            scores[chunk_id] = initial_score

        # BFS-like propagation along edges
        # Queue contains (chunk_id, score_to_propagate)
        queue = deque()
        for chunk_id in latest_chunk_ids:
            queue.append(chunk_id)

        visited = set(latest_chunk_ids)

        while queue:
            current_id = queue.popleft()
            current_score = scores[current_id]

            # Get outgoing edges (edges where current_id is the source)
            if current_id not in knowledge_graph:
                continue

            successors = list(knowledge_graph.successors(current_id))
            out_degree = len(successors)

            if out_degree == 0:
                continue

            # Split score evenly by out-degree
            score_per_successor = current_score / out_degree

            for successor_id in successors:
                if successor_id not in scores:
                    scores[successor_id] = 0.0

                # Accumulate score at successor
                scores[successor_id] += score_per_successor

                # Add to queue if not visited
                if successor_id not in visited:
                    visited.add(successor_id)
                    queue.append(successor_id)

        return scores

    def _reflection_propagation(
        self,
        forward_scores: dict[int, float],
        knowledge_graph: nx.DiGraph,
    ) -> dict[int, float]:
        """Phase 2: Reflection propagation in reverse direction (from earlier to later).

        Args:
            forward_scores: Scores from forward propagation
            knowledge_graph: Knowledge graph with directed edges
            chunk_map: Mapping from chunk ID to chunk object

        Returns:
            Dictionary mapping chunk ID to reflection score
        """
        # Start from all nodes that received forward scores
        starting_nodes = list(forward_scores.keys())

        if not starting_nodes:
            return {}

        reflection_scores = {}
        for node_id in starting_nodes:
            reflection_scores[node_id] = forward_scores[node_id]

        # Topological processing: peel layers like an onion
        # Find nodes with no incoming edges first, propagate, then remove them
        graph_copy = knowledge_graph.copy()

        # Build reverse graph for reflection (reverse edge directions)
        reverse_graph = graph_copy.reverse()

        # BFS-like propagation in reverse direction
        queue = deque()

        # Start from nodes with no incoming edges in reverse graph
        # (i.e., nodes with no outgoing edges in original graph)
        for node_id in starting_nodes:
            if node_id in reverse_graph and reverse_graph.out_degree(node_id) == 0:
                queue.append(node_id)

        visited = set(queue)

        while queue:
            current_id = queue.popleft()

            if current_id not in reflection_scores:
                continue

            current_score = reflection_scores[current_id]

            # Get successors in reverse graph (predecessors in original graph)
            if current_id not in reverse_graph:
                continue

            successors = list(reverse_graph.successors(current_id))
            out_degree_reverse = len(successors)

            if out_degree_reverse == 0:
                continue

            # Split score evenly by out-degree in reverse graph
            score_per_successor = current_score / out_degree_reverse

            for successor_id in successors:
                if successor_id not in reflection_scores:
                    reflection_scores[successor_id] = 0.0

                # Accumulate score
                reflection_scores[successor_id] += score_per_successor

                # Add to queue if not visited
                if successor_id not in visited:
                    visited.add(successor_id)
                    queue.append(successor_id)

        return reflection_scores
