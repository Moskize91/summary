"""Cognitive chunk data structure for working memory."""

from dataclasses import dataclass, field

from .fragment import SentenceId


@dataclass
class ChunkBatch:
    """A batch of extracted chunks with their relationships."""

    chunks: list["CognitiveChunk"]  # Extracted chunks (id=0, to be assigned)
    temp_ids: list[str]  # Temporary IDs corresponding to chunks
    links: list[dict]  # Raw link data: [{"from": ..., "to": ..., "strength": "critical/important/helpful"}]
    order_correct: bool  # Whether JSON key order was correct


@dataclass
class CognitiveChunk:
    """A cognitive chunk representing a unit of information in working memory.

    Inspired by Miller's "chunking" concept in cognitive psychology.
    """

    id: int
    generation: int
    sentence_id: SentenceId  # Primary sentence ID (first sentence, for ordering)
    label: str  # Short summary (5-15 chars) for quick scanning
    content: str  # Full content
    chunk_type: int  # Type of chunk: 1=user_focused, 2=book_coherence
    sentence_ids: list[SentenceId] = field(default_factory=list)  # All sentence IDs that comprise this chunk
    links: list[int] = field(default_factory=list)
    retention: str | None = None  # user_focused: verbatim/detailed/focused/relevant
    importance: str | None = None  # book_coherence: critical/important/helpful

    def __repr__(self) -> str:
        """Return a readable representation."""
        links_str = f" -> {self.links}" if self.links else ""
        type_str = "UF" if self.chunk_type == 1 else "BC"
        return f"Chunk({self.id}[{type_str}]: [{self.label}] {self.content[:50]}...{links_str})"
