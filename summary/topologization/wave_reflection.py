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

    def __init__(self, generation_decay_factor: float):
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
        """Select top M extra chunks using Wave Reflection algorithm.

        New semantics: Only select from historical chunks (exclude latest_chunk_ids).
        Latest chunks are managed separately in WorkingMemory.

        Args:
            all_chunks: All available chunks (including current fragment)
            knowledge_graph: Knowledge graph with edges
            latest_chunk_ids: IDs of chunks from the latest extraction (to use as starting points)
            capacity: Number of extra chunks to select from history

        Returns:
            Top M extra chunks selected from history (excluding latest chunks)
        """
        if not all_chunks:
            return []

        if not latest_chunk_ids:
            # No new chunks, return top capacity chunks from history
            return all_chunks[:capacity]

        # Separate current fragment chunks from historical chunks
        latest_chunk_ids_set = set(latest_chunk_ids)
        historical_chunks = [c for c in all_chunks if c.id not in latest_chunk_ids_set]

        if not historical_chunks:
            # No historical chunks available
            return []

        # Phase 1: Forward propagation (starting from latest chunks)
        forward_scores = self._forward_propagation(
            latest_chunk_ids=latest_chunk_ids,
            knowledge_graph=knowledge_graph,
        )

        if not forward_scores:
            # No reachable nodes, return top historical chunks by generation
            historical_chunks.sort(key=lambda c: c.generation, reverse=True)
            return historical_chunks[:capacity]

        # Phase 2: Reflection propagation
        reflection_scores = self._reflection_propagation(forward_scores=forward_scores, knowledge_graph=knowledge_graph)

        # Score only historical chunks (exclude latest chunks)
        candidate_scores = []
        for chunk in historical_chunks:
            reflection_score = reflection_scores.get(chunk.id, 0.0)

            # Apply generation decay to reflection score
            decayed_score = reflection_score * (self.generation_decay_factor**chunk.generation)

            candidate_scores.append((chunk, reflection_score, decayed_score))

        # Sort by decayed score (descending) and select top capacity
        candidate_scores.sort(key=lambda x: x[2], reverse=True)

        selected_chunks = [chunk for chunk, _, _ in candidate_scores[:capacity]]

        return selected_chunks

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
        # Start from all nodes that have forward scores (not just leaf nodes)
        queue = deque(starting_nodes)
        visited = set(starting_nodes)

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
