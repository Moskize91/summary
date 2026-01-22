from collections.abc import Awaitable, Callable

from .cognitive_chunk import ChunkBatch, CognitiveChunk


class WorkingMemory:
    """Manages a limited-capacity working memory of cognitive chunks.

    New semantics (after two-stage extraction refactor):
    - Holds current fragment chunks (all extracted chunks in this fragment)
    - Plus a limited number of "extra" chunks selected from history (capacity)
    - Capacity only applies to extra chunks, not current fragment chunks
    """

    def __init__(self, capacity: int, id_generator: Callable[[], Awaitable[int]]):
        """Initialize working memory.

        Args:
            capacity: Number of extra chunks to select from history
                     (default: 7, based on Miller's magic number)
                     Current fragment chunks do not count towards this limit
            id_generator: Async callable that returns next available chunk ID
        """
        self.capacity = capacity
        self._current_fragment_chunks: list[CognitiveChunk] = []  # Chunks from current fragment
        self._extra_chunks: list[CognitiveChunk] = []  # Extra chunks selected from history
        self._id_generator = id_generator
        self._generation = 0  # Tracks how many times we've added new chunks

    async def add_chunks_with_links(
        self, chunk_batch: ChunkBatch
    ) -> tuple[list[CognitiveChunk], list[tuple[int, int]]]:
        """Add new chunks with link processing to working memory.

        New behavior: Only assigns IDs and adds to current fragment chunks.
        Selection of extra chunks happens externally via wave_reflection.

        Args:
            chunk_batch: ChunkBatch from extractor

        Returns:
            Tuple of (added_chunks, edges) where edges is list of (from_id, to_id) tuples
        """
        # Create temp_id to chunk mapping
        temp_id_map = {}

        # Assign IDs and generation to new chunks
        for chunk, temp_id in zip(chunk_batch.chunks, chunk_batch.temp_ids):
            chunk.id = await self._id_generator()
            chunk.generation = self._generation
            temp_id_map[temp_id] = chunk

        # Add to current fragment chunks
        self._current_fragment_chunks.extend(chunk_batch.chunks)

        # Process links and build chunk.links arrays + collect edges
        edges = []
        for link in chunk_batch.links:
            from_ref = link["from"]
            to_ref = link["to"]
            # Strength field will be used when saving to database (in topologize.py)

            # Resolve "from" reference
            if isinstance(from_ref, str):
                # temp_id reference
                from_chunk = temp_id_map.get(from_ref)
                if from_chunk is None:
                    print(f"Warning: temp_id '{from_ref}' not found in extracted chunks")
                    continue
            else:
                # Working memory ID reference - find in all visible chunks
                from_chunk = None
                for chunk in self.get_chunks():
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
                # Working memory ID reference - find in all visible chunks
                to_chunk = None
                for chunk in self.get_chunks():
                    if chunk.id == to_ref:
                        to_chunk = chunk
                        break
                if to_chunk is None:
                    print(f"Warning: Working memory ID {to_ref} not found")
                    continue

            # Normalize edge direction: always from larger ID to smaller ID
            # AI's link direction is ignored - only the existence of the link matters
            if from_chunk.id > to_chunk.id:
                edge_from_id = from_chunk.id
                edge_to_id = to_chunk.id
            else:
                edge_from_id = to_chunk.id
                edge_to_id = from_chunk.id

            # Add link to the "to" chunk's links array (edge_to_id is linked FROM edge_from_id)
            for chunk in chunk_batch.chunks:
                if chunk.id == edge_to_id:
                    if edge_from_id not in chunk.links:
                        chunk.links.append(edge_from_id)
                    break

            # Also check extra chunks (they might be referenced)
            if edge_to_id not in [c.id for c in chunk_batch.chunks]:
                for chunk in self._extra_chunks:
                    if chunk.id == edge_to_id:
                        if edge_from_id not in chunk.links:
                            chunk.links.append(edge_from_id)
                        break

            # Collect edge for later addition to knowledge graph
            edges.append((edge_from_id, edge_to_id))

        return chunk_batch.chunks, edges

    def set_extra_chunks(self, extra_chunks: list[CognitiveChunk]) -> None:
        """Set the extra chunks selected from history.

        Called after wave_reflection selects top chunks.

        Args:
            extra_chunks: List of chunks selected from history (max capacity)
        """
        self._extra_chunks = extra_chunks

    def finalize_fragment(self) -> list[CognitiveChunk]:
        """Finalize current fragment and prepare for next fragment.

        Returns:
            List of chunks that were in the current fragment (for tracking)
        """
        finished_chunks = self._current_fragment_chunks.copy()
        self._current_fragment_chunks = []

        # Increment generation for next fragment
        self._generation += 1

        return finished_chunks

    def get_chunks(self) -> list[CognitiveChunk]:
        """Get current chunks in working memory.

        Returns:
            List of current fragment chunks + extra chunks
        """
        return self._current_fragment_chunks + self._extra_chunks

    def get_all_chunks_for_saving(self) -> list[CognitiveChunk]:
        """Get all chunks that should be saved (current fragment only).

        Returns:
            List of current fragment chunks (extra chunks are already saved)
        """
        return self._current_fragment_chunks.copy()

    def format_for_prompt(self, include_current_fragment: bool = True) -> str:
        """Format chunks for LLM prompt.

        Args:
            include_current_fragment: If True, include current fragment chunks.
                                     If False, only show extra chunks from history.

        Returns:
            Formatted string representation with [label] - content format
        """
        if include_current_fragment:
            chunks = self.get_chunks()
        else:
            chunks = self._extra_chunks

        if not chunks:
            return "(empty)"

        lines = []
        for chunk in chunks:
            lines.append(f"{chunk.id}. [{chunk.label}] - {chunk.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all chunks from working memory."""
        self._current_fragment_chunks.clear()
        self._extra_chunks.clear()
