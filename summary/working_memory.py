"""Working memory manager for cognitive chunks."""

from .cognitive_chunk import CognitiveChunk


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
