from .cognitive_chunk import ChunkBatch, CognitiveChunk


class WorkingMemory:
    """Manages a limited-capacity working memory of cognitive chunks.

    Implements a scoring system based on:
    - Freshness: newly added chunks get higher scores
    - References: chunks that are referenced more get higher scores
    """

    def __init__(self, capacity: int = 7):
        """Initialize working memory.

        Args:
            capacity: Maximum number of chunks to keep in working memory
                     (default: 7, based on Miller's magic number)
        """
        self.capacity = capacity
        self._chunks: list[CognitiveChunk] = []
        self._next_id = 1
        self._generation = 0  # Tracks how many times we've added new chunks

    def add_chunks(self, new_chunks: list[CognitiveChunk]) -> None:
        """Add new chunks to working memory and evict low-scoring chunks.

        Args:
            new_chunks: List of new chunks to add
        """
        # Increment generation first
        self._generation += 1

        # Assign IDs and generation to new chunks
        for chunk in new_chunks:
            chunk.id = self._next_id
            chunk.generation = self._generation
            self._next_id += 1

        # Merge new and existing chunks
        all_chunks = self._chunks + new_chunks

        # Calculate scores and keep top-k
        scored_chunks = [(chunk, self._calculate_score(chunk, all_chunks)) for chunk in all_chunks]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        self._chunks = [chunk for chunk, _ in scored_chunks[: self.capacity]]

    def add_chunks_with_links(self, chunk_batch: ChunkBatch) -> tuple[list[CognitiveChunk], list[tuple[int, int]]]:
        """Add new chunks with link processing to working memory.

        Args:
            chunk_batch: ChunkBatch from extractor

        Returns:
            Tuple of (added_chunks, edges) where edges is list of (from_id, to_id) tuples
        """
        # Increment generation first
        self._generation += 1

        # Create temp_id to chunk mapping
        temp_id_map = {}

        # Assign IDs and generation to new chunks
        for chunk, temp_id in zip(chunk_batch.chunks, chunk_batch.temp_ids):
            chunk.id = self._next_id
            chunk.generation = self._generation
            temp_id_map[temp_id] = chunk
            self._next_id += 1

        # Process links and build chunk.links arrays + collect edges
        edges = []
        for link in chunk_batch.links:
            from_ref = link["from"]
            to_ref = link["to"]

            # Resolve "from" reference
            if isinstance(from_ref, str):
                # temp_id reference
                from_chunk = temp_id_map.get(from_ref)
                if from_chunk is None:
                    print(f"Warning: temp_id '{from_ref}' not found in extracted chunks")
                    continue
            else:
                # Working memory ID reference - find in existing chunks
                from_chunk = None
                for chunk in self._chunks:
                    if chunk.id == from_ref:
                        from_chunk = chunk
                        break
                if from_chunk is None:
                    print(f"Warning: Working memory ID {from_ref} not found")
                    continue

            # Resolve "to" reference
            if isinstance(to_ref, str):
                # temp_id reference
                to_chunk = temp_id_map.get(to_ref)
                if to_chunk is None:
                    print(f"Warning: temp_id '{to_ref}' not found in extracted chunks")
                    continue
            else:
                # Working memory ID reference - find in existing chunks
                to_chunk = None
                for chunk in self._chunks:
                    if chunk.id == to_ref:
                        to_chunk = chunk
                        break
                if to_chunk is None:
                    print(f"Warning: Working memory ID {to_ref} not found")
                    continue

            # Normalize edge direction: always from later to earlier (based on sentence_id)
            # If sentence_id is the same, use chunk.id as tiebreaker
            if from_chunk.sentence_id > to_chunk.sentence_id or (
                from_chunk.sentence_id == to_chunk.sentence_id and from_chunk.id > to_chunk.id
            ):
                # from is later, to is earlier: edge should be from -> to (correct direction)
                edge_from_id = from_chunk.id
                edge_to_id = to_chunk.id
            else:
                # to is later, from is earlier: reverse edge direction
                edge_from_id = to_chunk.id
                edge_to_id = from_chunk.id

            # Add link to the "to" chunk's links array (edge_to_id is linked FROM edge_from_id)
            for chunk in chunk_batch.chunks:
                if chunk.id == edge_to_id:
                    if edge_from_id not in chunk.links:
                        chunk.links.append(edge_from_id)
                    break

            # Also check existing chunks in working memory
            if edge_to_id not in [c.id for c in chunk_batch.chunks]:
                for chunk in self._chunks:
                    if chunk.id == edge_to_id:
                        if edge_from_id not in chunk.links:
                            chunk.links.append(edge_from_id)
                        break

            # Collect edge for later addition to knowledge graph
            edges.append((edge_from_id, edge_to_id))

        # Merge new and existing chunks for working memory
        all_chunks = self._chunks + chunk_batch.chunks

        # Calculate scores and keep top-k
        scored_chunks = [(chunk, self._calculate_score(chunk, all_chunks)) for chunk in all_chunks]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        self._chunks = [chunk for chunk, _ in scored_chunks[: self.capacity]]

        return chunk_batch.chunks, edges

    def _calculate_score(self, chunk: CognitiveChunk, all_chunks: list[CognitiveChunk]) -> float:
        """Calculate importance score for a chunk.

        Score = freshness + reference_count

        Args:
            chunk: The chunk to score
            all_chunks: All candidate chunks

        Returns:
            Importance score
        """
        # Freshness: newer chunks get higher scores, decays with generation
        age = self._generation - (chunk.id // 100)  # Rough age estimate
        freshness = 1.0 / (1.0 + age)

        # Reference count: how many chunks link to this chunk
        reference_count = sum(1 for c in all_chunks if chunk.id in c.links)

        return freshness + reference_count

    def get_chunks(self) -> list[CognitiveChunk]:
        """Get current chunks in working memory.

        Returns:
            List of current chunks
        """
        return self._chunks.copy()

    def format_for_prompt(self) -> str:
        """Format chunks for LLM prompt.

        Returns:
            Formatted string representation with [label] - content format
        """
        if not self._chunks:
            return "(empty)"

        lines = []
        for chunk in self._chunks:
            lines.append(f"{chunk.id}. [{chunk.label}] - {chunk.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all chunks from working memory."""
        self._chunks.clear()
