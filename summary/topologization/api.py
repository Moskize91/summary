"""Public API for accessing topologization results from workspace."""
# pylint: disable=protected-access
# Note: Chunk and Snake classes intentionally access Topologization's protected members
# for lazy-loading data from the database. This is a friend-class design pattern.

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from .fragment import FragmentReader, SentenceId
from .graph import Graph


@dataclass
class ChunkEdge:
    """Represents a directed edge between two chunks in the knowledge graph.

    Attributes:
        from_chunk: Source chunk
        to_chunk: Target chunk
        strength: Link strength (critical/important/helpful)
        weight: Edge weight (computed from node attributes and link strength)
    """

    from_chunk: "Chunk"
    to_chunk: "Chunk"
    strength: str | None = None
    weight: float = 0.1


@dataclass
class SnakeEdge:
    """Represents a directed edge between two snakes in the snake graph.

    Attributes:
        from_snake: Source snake
        to_snake: Target snake
        weight: Total weight of chunk edges between these snakes
    """

    from_snake: "Snake"
    to_snake: "Snake"
    weight: float


@dataclass
class Chunk:
    """Knowledge graph chunk node.

    Represents a cognitive chunk with metadata and lazy-loaded content.
    """

    id: int
    generation: int
    sentence_id: SentenceId  # Primary sentence ID for ordering
    label: str
    content: str  # AI-generated summary content
    _topologization: "Topologization"  # Reference for lazy loading
    retention: str | None = None  # verbatim/detailed/focused/relevant
    importance: str | None = None  # critical/important/helpful
    tokens: int = 0  # Total token count of original source sentences
    weight: float = 0.0  # Node weight (computed from retention + importance)
    _sentence_ids: list[SentenceId] | None = field(default=None, repr=False)  # Lazy-loaded

    @property
    def sentence_ids(self) -> list[SentenceId]:
        """Get all sentence IDs that comprise this chunk.

        Returns:
            List of (chapter_id, fragment_id, sentence_index) tuples
        """
        if self._sentence_ids is None:
            # Query chunk_sentences table to get all sentences
            cursor = self._topologization._conn.execute(
                "SELECT chapter_id, fragment_id, sentence_index FROM chunk_sentences WHERE chunk_id = ? "
                "ORDER BY chapter_id, fragment_id, sentence_index",
                (self.id,),
            )
            self._sentence_ids = [(row[0], row[1], row[2]) for row in cursor]

        return self._sentence_ids

    def get_outgoing_edges(self) -> list[ChunkEdge]:
        """Get edges from this chunk to other chunks.

        Returns:
            List of ChunkEdge objects
        """
        cursor = self._topologization._conn.execute(
            "SELECT to_id, strength, weight FROM knowledge_edges WHERE from_id = ?",
            (self.id,),
        )
        edges_data = [(row[0], row[1], row[2]) for row in cursor]
        return [
            ChunkEdge(
                from_chunk=self,
                to_chunk=self._topologization.get_chunk(to_id),
                strength=strength,
                weight=weight,
            )
            for to_id, strength, weight in edges_data
        ]

    def get_incoming_edges(self) -> list[ChunkEdge]:
        """Get edges from other chunks to this chunk.

        Returns:
            List of ChunkEdge objects
        """
        cursor = self._topologization._conn.execute(
            "SELECT from_id, strength, weight FROM knowledge_edges WHERE to_id = ?",
            (self.id,),
        )
        edges_data = [(row[0], row[1], row[2]) for row in cursor]
        return [
            ChunkEdge(
                from_chunk=self._topologization.get_chunk(from_id),
                to_chunk=self,
                strength=strength,
                weight=weight,
            )
            for from_id, strength, weight in edges_data
        ]


@dataclass
class Snake:
    """Snake (thematic chain) node.

    Represents a sequence of related chunks forming a coherent narrative thread.
    """

    snake_id: int
    size: int
    first_label: str
    last_label: str
    _topologization: "Topologization"
    tokens: int = 0  # Total tokens in snake (sum of chunk tokens)
    weight: float = 0.0  # Total weight of snake (sum of chunk weights)
    _chunk_ids: list[int] | None = field(default=None, repr=False)  # Lazy-loaded

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

    def get_outgoing_edges(self) -> list[SnakeEdge]:
        """Get edges from this snake to other snakes.

        Returns:
            List of SnakeEdge objects with weights
        """
        cursor = self._topologization._conn.execute(
            "SELECT to_snake, weight FROM snake_edges WHERE from_snake = ?",
            (self.snake_id,),
        )
        edges_data = [(row[0], row[1]) for row in cursor]
        return [
            SnakeEdge(
                from_snake=self,
                to_snake=self._topologization.get_snake(to_id),
                weight=weight,
            )
            for to_id, weight in edges_data
        ]

    def get_incoming_edges(self) -> list[SnakeEdge]:
        """Get edges from other snakes to this snake.

        Returns:
            List of SnakeEdge objects with weights
        """
        cursor = self._topologization._conn.execute(
            "SELECT from_snake, weight FROM snake_edges WHERE to_snake = ?",
            (self.snake_id,),
        )
        edges_data = [(row[0], row[1]) for row in cursor]
        return [
            SnakeEdge(
                from_snake=self._topologization.get_snake(from_id),
                to_snake=self,
                weight=weight,
            )
            for from_id, weight in edges_data
        ]


class KnowledgeGraph(Graph[Chunk, ChunkEdge]):
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
        self._edge_tuples: list[tuple[int, int, str | None, float]] = []
        self._load_structure()

    def _load_structure(self):
        """Load node IDs and edges from database."""
        # Load all node IDs
        cursor = self._conn.execute("SELECT id FROM chunks ORDER BY id")
        self._node_ids = [row[0] for row in cursor]

        # Load all edges with weight
        cursor = self._conn.execute("SELECT from_id, to_id, strength, weight FROM knowledge_edges")
        self._edge_tuples = [(row[0], row[1], row[2], row[3]) for row in cursor]

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

    def get_edges(self) -> list[ChunkEdge]:
        """Get all edges as ChunkEdge objects.

        Returns:
            List of ChunkEdge objects
        """
        return [
            ChunkEdge(
                from_chunk=self._topologization.get_chunk(from_id),
                to_chunk=self._topologization.get_chunk(to_id),
                strength=strength,
                weight=weight,
            )
            for from_id, to_id, strength, weight in self._edge_tuples
        ]

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

        # Add all edges with weight
        for from_id, to_id, strength, weight in self._edge_tuples:
            g.add_edge(from_id, to_id, strength=strength, weight=weight)

        return g


class SnakeGraph(Graph[Snake, SnakeEdge]):
    """Snake graph implementation with lazy-loading.

    Implements the Graph protocol for snake-level graph.
    Scoped to a specific chapter and group.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        topologization: "Topologization",
        chapter_id: int,
        group_id: int,
    ):
        """Initialize snake graph for a specific chapter and group.

        Args:
            conn: SQLite database connection
            topologization: Parent Topologization instance
            chapter_id: Chapter ID to filter snakes
            group_id: Group ID to filter snakes
        """
        self._conn = conn
        self._topologization = topologization
        self._chapter_id = chapter_id
        self._group_id = group_id
        # Load graph structure at construction
        self._snake_ids: list[int] = []
        self._edge_tuples: list[tuple[int, int, float]] = []  # (from, to, weight)
        self._load_structure()

    def _load_structure(self):
        """Load snake IDs and edges from database for this chapter/group."""
        # Load snake IDs for this chapter and group
        cursor = self._conn.execute(
            "SELECT id FROM snakes WHERE chapter_id = ? AND group_id = ? ORDER BY id",
            (self._chapter_id, self._group_id),
        )
        self._snake_ids = [row[0] for row in cursor]

        # Load edges between snakes in this group
        if self._snake_ids:
            placeholders = ",".join("?" * len(self._snake_ids))
            cursor = self._conn.execute(
                f"""
                SELECT from_snake_id, to_snake_id, weight
                FROM snake_edges
                WHERE from_snake_id IN ({placeholders}) AND to_snake_id IN ({placeholders})
            """,
                self._snake_ids + self._snake_ids,
            )
            self._edge_tuples = [(row[0], row[1], row[2]) for row in cursor]

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

    def get_edges(self) -> list[SnakeEdge]:
        """Get all edges as SnakeEdge objects.

        Returns:
            List of SnakeEdge objects with weight
        """
        return [
            SnakeEdge(
                from_snake=self._topologization.get_snake(from_id),
                to_snake=self._topologization.get_snake(to_id),
                weight=weight,
            )
            for from_id, to_id, weight in self._edge_tuples
        ]

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

        # Add all edges with weight
        for from_id, to_id, weight in self._edge_tuples:
            g.add_edge(from_id, to_id, weight=weight)

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
        self.knowledge_graph: Graph[Chunk, ChunkEdge] = KnowledgeGraph(self._conn, self)
        # Note: snake_graph is now accessed per-group via get_snake_graph(chapter_id, group_id)

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
            (
                "SELECT id, generation, chapter_id, fragment_id, sentence_index, label, "
                "content, retention, importance, tokens, weight"
                " FROM chunks WHERE id = ?"
            ),
            (chunk_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Chunk {chunk_id} not found")

        return Chunk(
            id=row[0],
            generation=row[1],
            sentence_id=(row[2], row[3], row[4]),
            label=row[5],
            content=row[6],
            retention=row[7],
            importance=row[8],
            tokens=row[9],
            weight=row[10],
            _topologization=self,
        )

    def get_snake(self, snake_id: int) -> Snake:
        """Load snake from database.

        Args:
            snake_id: Global snake ID

        Returns:
            Snake object

        Raises:
            ValueError: If snake not found
        """
        cursor = self._conn.execute(
            (
                "SELECT id, chapter_id, group_id, local_snake_id, size, first_label, last_label, tokens, weight "
                "FROM snakes WHERE id = ?"
            ),
            (snake_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Snake {snake_id} not found")

        return Snake(
            snake_id=row[0],
            size=row[4],
            first_label=row[5],
            last_label=row[6],
            tokens=row[7],
            weight=row[8],
            _topologization=self,
        )

    def get_all_chapter_ids(self) -> list[int]:
        """Get list of all chapter IDs.

        Returns:
            Sorted list of chapter IDs
        """
        cursor = self._conn.execute(
            "SELECT DISTINCT chapter_id FROM fragment_groups ORDER BY chapter_id"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_group_ids_for_chapter(self, chapter_id: int) -> list[int]:
        """Get list of all group IDs for a specific chapter.

        Args:
            chapter_id: Chapter ID

        Returns:
            Sorted list of group IDs
        """
        cursor = self._conn.execute(
            "SELECT DISTINCT group_id FROM fragment_groups WHERE chapter_id = ? ORDER BY group_id",
            (chapter_id,),
        )
        return [row[0] for row in cursor.fetchall()]

    def get_snake_graph(self, chapter_id: int, group_id: int) -> Graph[Snake, SnakeEdge]:
        """Get snake graph for a specific chapter and group.

        Args:
            chapter_id: Chapter ID
            group_id: Group ID

        Returns:
            Snake graph for the specified group
        """
        return SnakeGraph(self._conn, self, chapter_id, group_id)

    def close(self):
        """Close database connection."""
        self._conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()
