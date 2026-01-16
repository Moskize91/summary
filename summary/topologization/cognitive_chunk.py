"""Cognitive chunk data structure for working memory."""

from dataclasses import dataclass, field

from .storage import SentenceId


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
    sentence_ids: list[SentenceId] = field(default_factory=list)  # All sentence IDs that comprise this chunk
    links: list[int] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return a readable representation."""
        links_str = f" -> {self.links}" if self.links else ""
        return f"Chunk({self.id}: [{self.label}] {self.content[:50]}...{links_str})"
