"""Cognitive chunk data structure for working memory."""

from dataclasses import dataclass, field


@dataclass
class CognitiveChunk:
    """A cognitive chunk representing a unit of information in working memory.

    Inspired by Miller's "chunking" concept in cognitive psychology.
    """

    id: int
    content: str
    links: list[int] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return a readable representation."""
        links_str = f" -> {self.links}" if self.links else ""
        return f"Chunk({self.id}: {self.content[:50]}...{links_str})"
