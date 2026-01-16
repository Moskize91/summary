"""Public API for accessing topologization results from workspace."""
# pylint: disable=protected-access
# Note: Chunk and Snake classes intentionally access Topologization's protected members
# for lazy-loading data from the database. This is a friend-class design pattern.

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from .graph import Graph
from .storage import FragmentReader, SentenceId


@dataclass
class Chunk:
    """Knowledge graph chunk node.

    Represents a cognitive chunk with metadata and lazy-loaded content.
    """

    id: int
    generation: int
    sentence_id: SentenceId  # Primary sentence ID for ordering
    label: str
    _topologization: "Topologization" = field(repr=False)  # Reference for lazy loading
    _content: str | None = field(default=None, repr=False)  # Lazy-loaded
    _sentence_ids: list[SentenceId] | None = field(default=None, repr=False)  # Lazy-loaded

    @property
    def content(self) -> str:
        """Lazy-load full content by reading all associated sentences.

        Returns:
            Full text content of this chunk
        """
        if self._content is None:
            # Query chunk_sentences table to get all sentences
            cursor = self._topologization._conn.execute(
                "SELECT fragment_id, sentence_index FROM chunk_sentences WHERE chunk_id = ? "
                "ORDER BY fragment_id, sentence_index",
                (self.id,),
            )
            sentence_ids = [(row[0], row[1]) for row in cursor]

            # Load all sentences and join
            sentences = [self._topologization.get_sentence_text(sid) for sid in sentence_ids]
            self._content = " ".join(sentences)
            self._sentence_ids = sentence_ids

        return self._content

    @property
    def sentence_ids(self) -> list[SentenceId]:
        """Get all sentence IDs that comprise this chunk.

        Returns:
            List of (fragment_id, sentence_index) tuples
        """
        if self._sentence_ids is None:
            # Trigger content loading which also loads sentence_ids
            _ = self.content
        return self._sentence_ids or []

    def get_outgoing_edges(self) -> list[int]:
        """Get IDs of chunks this chunk links to.

        Returns:
            List of chunk IDs
        """
        cursor = self._topologization._conn.execute(
            "SELECT to_id FROM knowledge_edges WHERE from_id = ?",
            (self.id,),
        )
        return [row[0] for row in cursor]

    def get_incoming_edges(self) -> list[int]:
        """Get IDs of chunks that link to this chunk.

        Returns:
            List of chunk IDs
        """
        cursor = self._topologization._conn.execute(
            "SELECT from_id FROM knowledge_edges WHERE to_id = ?",
            (self.id,),
        )
        return [row[0] for row in cursor]


@dataclass
class Snake:
    """Snake (thematic chain) node.

    Represents a sequence of related chunks forming a coherent narrative thread.
    """

    snake_id: int
    size: int
    first_label: str
    last_label: str
    _topologization: "Topologization" = field(repr=False)
    _summary: str | None = field(default=None, repr=False)  # Lazy-loaded
    _chunk_ids: list[int] | None = field(default=None, repr=False)  # Lazy-loaded

    @property
    def summary(self) -> str:
        """Lazy-load LLM-generated summary from database.

        Returns:
            Summary text
        """
        if self._summary is None:
            cursor = self._topologization._conn.execute(
                "SELECT summary FROM snake_summaries WHERE snake_id = ?",
                (self.snake_id,),
            )
            row = cursor.fetchone()
            self._summary = row[0] if row else ""
        return self._summary or ""

    @property
    def chunk_ids(self) -> list[int]:
        """Lazy-load chunk IDs from database.

        Returns:
            List of chunk IDs in snake order
        """
        if self._chunk_ids is None:
            cursor = self._topologization._conn.execute(
                "SELECT chunk_id FROM snake_chunks WHERE snake_id = ? ORDER BY position",
                (self.snake_id,),
            )
            self._chunk_ids = [row[0] for row in cursor]
        return self._chunk_ids

    def get_chunks(self) -> list[Chunk]:
        """Get all chunks in this snake.

        Returns:
            List of Chunk objects in snake order
        """
        return [self._topologization.get_chunk(cid) for cid in self.chunk_ids]

    def get_outgoing_edges(self) -> list[int]:
        """Get IDs of snakes this snake links to.

        Returns:
            List of snake IDs
        """
        cursor = self._topologization._conn.execute(
            "SELECT to_snake FROM snake_edges WHERE from_snake = ?",
            (self.snake_id,),
        )
        return [row[0] for row in cursor]

    def get_incoming_edges(self) -> list[int]:
        """Get IDs of snakes that link to this snake.

        Returns:
            List of snake IDs
        """
        cursor = self._topologization._conn.execute(
            "SELECT from_snake FROM snake_edges WHERE to_snake = ?",
            (self.snake_id,),
        )
        return [row[0] for row in cursor]


class KnowledgeGraph(Graph[Chunk]):
    """Knowledge graph implementation with lazy-loading.

    Implements the Graph protocol for chunk-level knowledge graph.
    """

    def __init__(self, conn: sqlite3.Connection, topologization: "Topologization"):
        """Initialize knowledge graph.

        Args:
            conn: SQLite database connection
            topologization: Parent Topologization instance
        """
        self._conn = conn
        self._topologization = topologization
        # Load graph structure (node IDs + edges) at construction
        self._node_ids: list[int] = []
        self._edges: list[tuple[int, int]] = []
        self._load_structure()

    def _load_structure(self):
        """Load node IDs and edges from database."""
        # Load all node IDs
        cursor = self._conn.execute("SELECT id FROM chunks ORDER BY id")
        self._node_ids = [row[0] for row in cursor]

        # Load all edges
        cursor = self._conn.execute("SELECT from_id, to_id FROM knowledge_edges")
        self._edges = [(row[0], row[1]) for row in cursor]

    def __iter__(self) -> Iterator[Chunk]:
        """Iterate all chunks (lazy-load content on demand).

        Yields:
            Chunk objects
        """
        for chunk_id in self._node_ids:
            yield self._topologization.get_chunk(chunk_id)

    def get_node(self, chunk_id: int) -> Chunk:
        """Get specific chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk object
        """
        return self._topologization.get_chunk(chunk_id)

    def get_edges(self) -> list[tuple[int, int]]:
        """Get all edges as (from_id, to_id) pairs.

        Returns:
            List of edge tuples
        """
        return self._edges.copy()

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph for visualization.

        Returns:
            NetworkX directed graph
        """
        g = nx.DiGraph()

        # Add all nodes with attributes
        for chunk in self:
            g.add_node(
                chunk.id,
                generation=chunk.generation,
                sentence_id=chunk.sentence_id,
                label=chunk.label,
                # Don't add content to save memory
            )

        # Add all edges
        for from_id, to_id in self._edges:
            g.add_edge(from_id, to_id)

        return g


class SnakeGraph(Graph[Snake]):
    """Snake graph implementation with lazy-loading.

    Implements the Graph protocol for snake-level graph.
    """

    def __init__(self, conn: sqlite3.Connection, topologization: "Topologization"):
        """Initialize snake graph.

        Args:
            conn: SQLite database connection
            topologization: Parent Topologization instance
        """
        self._conn = conn
        self._topologization = topologization
        # Load graph structure at construction
        self._snake_ids: list[int] = []
        self._edges: list[tuple[int, int, int]] = []  # (from, to, count)
        self._load_structure()

    def _load_structure(self):
        """Load snake IDs and edges from database."""
        # Load all snake IDs
        cursor = self._conn.execute("SELECT snake_id FROM snakes ORDER BY snake_id")
        self._snake_ids = [row[0] for row in cursor]

        # Load all edges with counts
        cursor = self._conn.execute("SELECT from_snake, to_snake, internal_edge_count FROM snake_edges")
        self._edges = [(row[0], row[1], row[2]) for row in cursor]

    def __iter__(self) -> Iterator[Snake]:
        """Iterate all snakes (lazy-load summaries on demand).

        Yields:
            Snake objects
        """
        for snake_id in self._snake_ids:
            yield self._topologization.get_snake(snake_id)

    def get_node(self, snake_id: int) -> Snake:
        """Get specific snake by ID.

        Args:
            snake_id: Snake ID

        Returns:
            Snake object
        """
        return self._topologization.get_snake(snake_id)

    def get_edges(self) -> list[tuple[int, int, int]]:
        """Get all edges as (from_id, to_id, count) tuples.

        Returns:
            List of edge tuples with internal edge counts
        """
        return self._edges.copy()

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph for visualization.

        Returns:
            NetworkX directed graph
        """
        g = nx.DiGraph()

        # Add all nodes with attributes
        for snake in self:
            g.add_node(
                snake.snake_id,
                size=snake.size,
                first_label=snake.first_label,
                last_label=snake.last_label,
                # Summary loaded on demand
            )

        # Add all edges
        for from_id, to_id, count in self._edges:
            g.add_edge(from_id, to_id, internal_edge_count=count)

        return g


class Topologization:
    """Main API for accessing topologization results from workspace.

    Provides access to knowledge graph and snake graph with lazy-loading.
    """

    def __init__(self, workspace_path: Path):
        """Load topologization from existing workspace.

        Args:
            workspace_path: Path to workspace directory containing fragments/ and database.db
        """
        self.workspace_path = workspace_path
        self._db_path = workspace_path / "database.db"

        # Initialize database connection
        if not self._db_path.exists():
            raise FileNotFoundError(f"Database not found: {self._db_path}")

        self._conn = sqlite3.connect(self._db_path)

        # Initialize fragment reader
        self._fragment_reader = FragmentReader(workspace_path)

        # Create graph objects (implement Graph protocol)
        self.knowledge_graph: Graph[Chunk] = KnowledgeGraph(self._conn, self)
        self.snake_graph: Graph[Snake] = SnakeGraph(self._conn, self)

    def get_sentence_text(self, sentence_id: SentenceId) -> str:
        """Lazy-load sentence text from fragment.json.

        Args:
            sentence_id: (fragment_id, sentence_index) tuple

        Returns:
            Sentence text string
        """
        return self._fragment_reader.get_sentence(sentence_id)

    def get_chunk(self, chunk_id: int) -> Chunk:
        """Load chunk from database with lazy-loaded content.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk object

        Raises:
            ValueError: If chunk not found
        """
        cursor = self._conn.execute(
            "SELECT id, generation, fragment_id, sentence_index, label FROM chunks WHERE id = ?",
            (chunk_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Chunk {chunk_id} not found")

        return Chunk(
            id=row[0],
            generation=row[1],
            sentence_id=(row[2], row[3]),
            label=row[4],
            _topologization=self,
        )

    def get_snake(self, snake_id: int) -> Snake:
        """Load snake from database.

        Args:
            snake_id: Snake ID

        Returns:
            Snake object

        Raises:
            ValueError: If snake not found
        """
        cursor = self._conn.execute(
            "SELECT snake_id, size, first_label, last_label FROM snakes WHERE snake_id = ?",
            (snake_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Snake {snake_id} not found")

        return Snake(
            snake_id=row[0],
            size=row[1],
            first_label=row[2],
            last_label=row[3],
            _topologization=self,
        )

    def close(self):
        """Close database connection."""
        self._conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()
