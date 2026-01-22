"""Topologization pipeline: incremental knowledge graph building and snake detection."""

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .api import Chunk, Graph, Snake

import networkx as nx
from tiktoken import Encoding

from ..llm import LLM
from . import database
from .chunk_extraction import ChunkExtractor
from .cognitive_chunk import CognitiveChunk
from .fragment import FragmentReader, FragmentWriter, SentenceId
from .fragment_grouping import group_fragments_by_chapter
from .graph_weights import add_weights_to_graph
from .snake_detector import SnakeDetector, split_connected_components
from .snake_graph_builder import SnakeGraphBuilder
from .text_fragmenter import TextFragmenter
from .wave_reflection import WaveReflection
from .working_memory import WorkingMemory

_GENERATION_DECAY_FACTOR = 0.5
_MIN_SNAKE_SIZE = 2


class ReadonlyTopologization:
    """Read-only access to topologization workspace.

    Provides query and access methods for reading existing workspace data
    without modification capabilities.
    """

    def __init__(self, workspace_path: Path):
        """Load existing workspace for read-only access.

        Args:
            workspace_path: Path to workspace directory containing fragments/ and database.db
        """
        # Validate workspace exists
        if not workspace_path.exists():
            raise FileNotFoundError(f"Workspace not found: {workspace_path}")

        db_path = workspace_path / "database.db"
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        # Setup connection and readers
        self._setup_connection(workspace_path, db_path)

    def _setup_connection(self, workspace_path: Path, db_path: Path):
        """Setup database connection and initialize readers.

        Args:
            workspace_path: Path to workspace directory
            db_path: Path to database file
        """
        self.workspace_path = workspace_path
        self._db_path = db_path
        self._conn = sqlite3.connect(self._db_path)

        # Initialize fragment reader
        self._fragment_reader = FragmentReader(workspace_path)

        # Create knowledge graph accessor (lazy-loading from database)
        from .api import KnowledgeGraph

        self.knowledge_graph = KnowledgeGraph(self._conn, self)

    def get_chunk(self, chunk_id: int) -> "Chunk":
        """Load chunk from database.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk object

        Raises:
            ValueError: If chunk not found
        """
        from .api import Chunk

        cursor = self._conn.execute(
            (
                "SELECT id, generation, chapter_id, fragment_id, sentence_index, label, content, "
                "retention, importance, tokens, weight FROM chunks WHERE id = ?"
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

    def get_snake(self, snake_id: int) -> "Snake":
        """Load snake from database.

        Args:
            snake_id: Global snake ID

        Returns:
            Snake object

        Raises:
            ValueError: If snake not found
        """
        from .api import Snake

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
        cursor = self._conn.execute("SELECT DISTINCT chapter_id FROM fragment_groups ORDER BY chapter_id")
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

    def get_snake_graph(self, chapter_id: int, group_id: int) -> "Graph":
        """Get snake graph for a specific chapter and group.

        Args:
            chapter_id: Chapter ID
            group_id: Group ID

        Returns:
            Snake graph for the specified group
        """
        from .api import SnakeGraph

        return SnakeGraph(self._conn, self, chapter_id, group_id)

    def get_sentence_text(self, sentence_id: SentenceId) -> str:
        """Lazy-load sentence text from fragment.json.

        Args:
            sentence_id: (chapter_id, fragment_id, sentence_index) tuple

        Returns:
            Sentence text string
        """
        return self._fragment_reader.get_sentence(sentence_id)

    def close(self):
        """Close database connection."""
        self._conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *_):
        """Context manager exit."""
        self.close()


class Topologization(ReadonlyTopologization):
    """Main class for topologization: building and accessing knowledge graphs.

    Supports two modes:
    1. Builder mode: Create with __init__, then call load() for each chapter incrementally
    2. Reader mode: Use from_workspace() to load existing workspace for reading only
    """

    def __init__(
        self,
        workspace_path: Path,
        llm: LLM,
        encoding: Encoding,
        max_fragment_tokens: int = 800,
        working_memory_capacity: int = 7,
        snake_tokens: int = 700,
        group_tokens_count: int = 9600,
    ):
        """Create or continue Topologization workspace for incremental chapter loading.

        Args:
            workspace_path: Directory to store fragments + database
            llm: LLM instance for extraction
            encoding: Token encoding
            max_fragment_tokens: Maximum tokens per fragment
            working_memory_capacity: Number of extra chunks in working memory
            generation_decay_factor: Decay factor for Wave Reflection
            min_cluster_size: Minimum snake size
            snake_tokens: Maximum tokens per snake
            group_tokens_count: Maximum tokens per fragment group
        """
        print("=" * 60)
        print("=== Initializing Topologization ===")
        print("=" * 60)
        print(f"Workspace: {workspace_path}")
        print(f"Working memory capacity: {working_memory_capacity}\n")

        # Setup workspace: create if database doesn't exist, continue if it does
        db_path = workspace_path / "database.db"

        if not db_path.exists():
            # Create new workspace
            print("Creating new workspace...")
            workspace_path.mkdir(parents=True, exist_ok=True)
            (workspace_path / "fragments").mkdir(exist_ok=True)

            # Initialize database
            conn = database.initialize_database(db_path)
            conn.close()
        else:
            # Continue from existing workspace
            print("Continuing from existing workspace...")

        # Call parent to setup connection and readers
        super().__init__(workspace_path)

        # Save build parameters
        self._llm = llm
        self._encoding = encoding
        self._max_fragment_tokens = max_fragment_tokens
        self._working_memory_capacity = working_memory_capacity
        self._snake_tokens = snake_tokens
        self._group_tokens_count = group_tokens_count

        # Calculate next IDs from database
        cursor = self._conn.execute("SELECT MAX(chapter_id) FROM fragment_groups")
        row = cursor.fetchone()
        max_chapter_id = row[0] if row else None
        self._next_chapter_id = (max_chapter_id + 1) if max_chapter_id is not None else 0

        cursor = self._conn.execute("SELECT MAX(id) FROM chunks")
        row = cursor.fetchone()
        max_chunk_id = row[0] if row else None
        self._next_chunk_id = (max_chunk_id + 1) if max_chunk_id is not None else 0

        print(f"Next chapter ID: {self._next_chapter_id}")
        print(f"Next chunk ID: {self._next_chunk_id}\n")

        # Extraction guidance cache (for intention changes)
        self._extraction_guidance: str | None = None
        self._last_intention: str | None = None

    async def load(self, sentences: Iterable[tuple[int, str]], intention: str) -> int:
        """Load one chapter and return its chapter_id.

        Args:
            sentences: Iterable of (token_count, sentence_text) for this chapter
            intention: User's reading intention/goal for extraction guidance

        Returns:
            Chapter ID (auto-incremented)
        """
        # Check if intention changed, regenerate guidance if needed
        if self._extraction_guidance is None or self._last_intention != intention:
            print("\n=== Meta-Prompt: Generating Extraction Guidance ===")
            self._extraction_guidance = await _generate_extraction_guidance(intention, self._llm)
            self._last_intention = intention

        # Get chapter ID and increment
        chapter_id = self._next_chapter_id
        self._next_chapter_id += 1

        print(f"\n{'=' * 60}")
        print(f"=== Loading Chapter {chapter_id} ===")
        print(f"{'=' * 60}")

        # Initialize fragment writer for this load session
        fragment_writer = FragmentWriter(self.workspace_path)

        # Load knowledge graph from database
        knowledge_graph_nx = _load_graph_from_database(self._conn)

        # Reset working memory (per-chapter independence)
        working_memory = WorkingMemory(capacity=self._working_memory_capacity)
        working_memory._next_id = self._next_chunk_id  # Set starting chunk ID
        # Note: generation resets to 0 automatically in new WorkingMemory

        # Create components for this chapter
        wave_reflection = WaveReflection(generation_decay_factor=_GENERATION_DECAY_FACTOR)
        extractor = ChunkExtractor(self._llm, self._extraction_guidance)

        # Start this chapter in fragment writer
        fragment_writer.start_chapter(chapter_id)

        # Extract knowledge graph for this chapter
        chapter_chunks = await self._extract_chapter_knowledge_graph(
            chapter_id=chapter_id,
            sentences=sentences,
            extractor=extractor,
            working_memory=working_memory,
            wave_reflection=wave_reflection,
            fragment_writer=fragment_writer,
            knowledge_graph_nx=knowledge_graph_nx,
        )

        # Update next_chunk_id for next chapter
        self._next_chunk_id = working_memory._next_id

        # Save chunks and edges to database
        print(f"\nSaving chapter {chapter_id} to database...")
        _save_knowledge_graph(self._conn, knowledge_graph_nx, chapter_chunks)

        # Fragment grouping (only this chapter)
        print(f"\n{'=' * 60}")
        print(f"=== Fragment Grouping (Chapter {chapter_id}) ===")
        print(f"{'=' * 60}")
        print(f"Group token limit: {self._group_tokens_count}")

        fragment_groups = group_fragments_by_chapter(self._conn, self.workspace_path, self._group_tokens_count)

        # Filter to only this chapter's groups
        chapter_groups = [g for g in fragment_groups if g.chapter_id == chapter_id]
        print(f"\nCreated {len(chapter_groups)} fragment groups:")
        for group in chapter_groups:
            print(f"  Chapter {group.chapter_id}, Group {group.group_id}: {len(group.fragment_ids)} fragments")

        # Save fragment groups to database
        _save_fragment_groups(self._conn, chapter_groups)

        # Snake detection (only this chapter)
        print(f"\n{'=' * 60}")
        print(f"=== Thematic Chain Detection (Chapter {chapter_id}) ===")
        print(f"{'=' * 60}")

        # Create config object for snake analysis
        group_snakes = _analyze_snakes_by_groups(
            self._conn,
            knowledge_graph_nx,
            chapter_groups,
            min_cluster_size=_MIN_SNAKE_SIZE,
            snake_tokens=self._snake_tokens,
        )

        # Save snakes to database
        print("\nSaving snakes to database...")
        _save_snakes_by_groups(self._conn, group_snakes, knowledge_graph_nx)

        # Print chapter summary
        total_snakes = sum(len(snakes_list) for snakes_list, _ in group_snakes.values())
        print(f"\n{'=' * 60}")
        print(f"=== Chapter {chapter_id} Complete ===")
        print(f"{'=' * 60}")
        print(f"Chunks: {len(chapter_chunks)}")
        print(f"Snakes: {total_snakes}")

        return chapter_id

    async def _extract_chapter_knowledge_graph(
        self,
        chapter_id: int,
        sentences: Iterable[tuple[int, str]],
        extractor: ChunkExtractor,
        working_memory: WorkingMemory,
        wave_reflection: WaveReflection,
        fragment_writer: FragmentWriter,
        knowledge_graph_nx: nx.DiGraph,
    ) -> list[CognitiveChunk]:
        """Extract knowledge graph for a single chapter.

        Args:
            chapter_id: Chapter ID being processed
            sentences: Iterable of (token_count, sentence_text) for this chapter
            extractor: Chunk extractor
            working_memory: Working memory (fresh for this chapter)
            wave_reflection: Wave reflection algorithm
            fragment_writer: Fragment writer for this session
            knowledge_graph_nx: Knowledge graph (loaded from database)

        Returns:
            List of all chunks extracted from this chapter
        """
        # Create fragmenter for this chapter (wraps single chapter as iterable)
        fragmenter = TextFragmenter(fragment_writer, self._encoding, self._max_fragment_tokens)

        chapter_chunks: list[CognitiveChunk] = []
        fragment_count = 0

        # Fragment this chapter's sentences
        for fragment_with_sentences in fragmenter.stream_fragments([sentences]):
            fragment_count += 1

            print(f"Processing chapter {chapter_id}, fragment {fragment_count}...")

            # === Stage 1: Extract user-focused chunks ===
            user_focused_result, fragment_summary = await extractor.extract_user_focused(
                fragment_with_sentences.text,
                working_memory,
                fragment_with_sentences.sentence_ids,
                fragment_with_sentences.sentence_texts,
                fragment_with_sentences.sentence_token_counts,
            )

            if user_focused_result is None:
                print(f"Warning: User-focused extraction failed for chapter {chapter_id}, fragment {fragment_count}")
                continue

            # Store fragment summary
            if fragment_summary:
                fragment_writer.set_summary(fragment_summary)

            # Add user-focused chunks to working memory and assign IDs
            user_focused_chunks, user_focused_edges = working_memory.add_chunks_with_links(user_focused_result)

            # Add user-focused chunks to knowledge graph
            for chunk in user_focused_chunks:
                knowledge_graph_nx.add_node(
                    chunk.id,
                    generation=chunk.generation,
                    sentence_id=chunk.sentence_id,
                    label=chunk.label,
                    content=chunk.content,
                    retention=chunk.retention,
                    importance=chunk.importance,
                    tokens=chunk.tokens,
                )

            # Add user-focused edges to knowledge graph with strength
            for from_id, to_id in user_focused_edges:
                strength = _find_edge_strength(
                    user_focused_result.links, from_id, to_id, user_focused_chunks, user_focused_result.temp_ids
                )
                knowledge_graph_nx.add_edge(from_id, to_id, strength=strength)

            chapter_chunks.extend(user_focused_chunks)

            # === Stage 2: Extract book-coherence chunks ===
            book_coherence_result = await extractor.extract_book_coherence(
                fragment_with_sentences.text,
                working_memory,
                user_focused_chunks,
                fragment_with_sentences.sentence_ids,
                fragment_with_sentences.sentence_texts,
                fragment_with_sentences.sentence_token_counts,
            )

            if book_coherence_result is not None and book_coherence_result.chunks:
                # Process importance_annotations: update Stage 1 chunks with importance
                if book_coherence_result.importance_annotations:
                    for annotation in book_coherence_result.importance_annotations:
                        chunk_id = annotation.get("chunk_id")
                        importance = annotation.get("importance")

                        # Find the chunk in user_focused_chunks and update its importance
                        for chunk in user_focused_chunks:
                            if chunk.id == chunk_id:
                                chunk.importance = importance
                                break

                # Add book-coherence chunks to working memory and assign IDs
                book_coherence_chunks, book_coherence_edges = working_memory.add_chunks_with_links(
                    book_coherence_result
                )

                # Add book-coherence chunks to knowledge graph
                for chunk in book_coherence_chunks:
                    knowledge_graph_nx.add_node(
                        chunk.id,
                        generation=chunk.generation,
                        sentence_id=chunk.sentence_id,
                        label=chunk.label,
                        content=chunk.content,
                        retention=chunk.retention,
                        importance=chunk.importance,
                        tokens=chunk.tokens,
                    )

                # Add book-coherence edges to knowledge graph with strength
                for from_id, to_id in book_coherence_edges:
                    strength = _find_edge_strength(
                        book_coherence_result.links,
                        from_id,
                        to_id,
                        book_coherence_chunks,
                        book_coherence_result.temp_ids,
                    )
                    knowledge_graph_nx.add_edge(from_id, to_id, strength=strength)

                chapter_chunks.extend(book_coherence_chunks)

            # === Update working memory with wave reflection ===
            # Get all chunks from current fragment (both stages)
            current_fragment_chunk_ids = [c.id for c in working_memory.get_all_chunks_for_saving()]

            # Select extra chunks from history using wave reflection
            # Note: Only select from chunks within this chapter (chapter independence)
            extra_chunks = wave_reflection.select_top_chunks(
                all_chunks=chapter_chunks,  # Only this chapter's chunks
                knowledge_graph=knowledge_graph_nx,
                latest_chunk_ids=current_fragment_chunk_ids,
                capacity=working_memory.capacity,
            )

            # Set extra chunks and finalize fragment
            working_memory.set_extra_chunks(extra_chunks)
            working_memory.finalize_fragment()

        # Finalize fragment writing for this chapter
        fragmenter.finalize()

        # Compute and add weights to knowledge graph for this chapter's chunks
        print(f"\nComputing node and edge weights for chapter {chapter_id}...")
        add_weights_to_graph(knowledge_graph_nx)

        print(f"\nChapter {chapter_id} extraction complete:")
        print(f"  Chunks: {len(chapter_chunks)}")
        print(f"  Fragments processed: {fragment_count}")

        return chapter_chunks


# ===== Internal helper functions =====


def _load_graph_from_database(conn: sqlite3.Connection) -> nx.DiGraph:
    """Load knowledge graph from database.

    Args:
        conn: SQLite database connection

    Returns:
        NetworkX directed graph with all nodes and edges
    """
    graph = nx.DiGraph()

    # Load all nodes
    cursor = conn.execute(
        "SELECT id, generation, chapter_id, fragment_id, sentence_index, "
        "label, content, retention, importance, tokens, weight FROM chunks"
    )
    for row in cursor:
        graph.add_node(
            row[0],  # id
            generation=row[1],
            sentence_id=(row[2], row[3], row[4]),  # (chapter_id, fragment_id, sentence_index)
            label=row[5],
            content=row[6],
            retention=row[7],
            importance=row[8],
            tokens=row[9],
            weight=row[10],
        )

    # Load all edges
    cursor = conn.execute("SELECT from_id, to_id, strength, weight FROM knowledge_edges")
    for row in cursor:
        graph.add_edge(row[0], row[1], strength=row[2], weight=row[3])

    return graph


async def _generate_extraction_guidance(
    intention: str,
    llm: LLM,
) -> str:
    """Generate extraction guidance from user intention using meta prompt.

    Args:
        intention: User's reading intention/goal
        llm: LLM instance

    Returns:
        Generated extraction guidance string

    Raises:
        RuntimeError: If guidance generation fails
    """
    print("Generating extraction guidance from intention...")

    # Find prompt template internally (relative to summary/data/)
    intention_prompt_file = Path(__file__).parent.parent / "data" / "topologization" / "chunk_extraction.jinja"
    system_prompt = llm.load_system_prompt(
        intention_prompt_file,
        intention=intention,
    )
    response = await llm.request(
        system_prompt=system_prompt,
        user_message=intention,
        temperature=0.3,
    )
    if not response:
        raise RuntimeError(
            "Failed to generate extraction guidance from intention. "
            "The meta-prompt LLM call did not return a valid response."
        )

    guidance = response.strip()
    if not guidance:
        raise RuntimeError(
            "Generated extraction guidance is empty. Please check the intention prompt template and user intention."
        )

    print(f"✓ Extraction guidance generated ({len(guidance)} characters)")
    return guidance


def _save_knowledge_graph(
    conn: sqlite3.Connection,
    knowledge_graph: nx.DiGraph,
    all_chunks: list[CognitiveChunk],
):
    """Save knowledge graph to database.

    Args:
        conn: Database connection
        knowledge_graph: Knowledge graph
        all_chunks: All chunks
    """
    # Save chunks with retention/importance metadata and computed weight
    for chunk in all_chunks:
        # Use the matched sentence IDs from source_sentences
        sentence_ids = chunk.sentence_ids if chunk.sentence_ids else [chunk.sentence_id]

        # Get computed weight from knowledge graph node
        node_weight = knowledge_graph.nodes[chunk.id].get("weight", 0.0)

        database.insert_chunk(
            conn,
            chunk.id,
            chunk.generation,
            chunk.sentence_id,
            chunk.label,
            chunk.content,  # AI-generated summary
            sentence_ids,
            retention=chunk.retention,
            importance=chunk.importance,
            tokens=chunk.tokens,
            weight=node_weight,
        )

    # Save edges with strength and weight metadata
    for from_id, to_id in knowledge_graph.edges():
        edge_data = knowledge_graph.edges[from_id, to_id]
        strength = edge_data.get("strength")
        weight = edge_data.get("weight", 0.1)
        database.insert_edge(conn, from_id, to_id, strength=strength, weight=weight)


def _save_fragment_groups(conn: sqlite3.Connection, fragment_groups: list):
    """Save fragment groups to database.

    Args:
        conn: Database connection
        fragment_groups: List of GroupInfo objects
    """
    cursor = conn.cursor()

    # Insert fragment group memberships
    for group in fragment_groups:
        for fragment_id in group.fragment_ids:
            cursor.execute(
                """
                INSERT INTO fragment_groups (chapter_id, group_id, fragment_id)
                VALUES (?, ?, ?)
            """,
                (group.chapter_id, group.group_id, fragment_id),
            )

    conn.commit()


def _analyze_snakes_by_groups(
    conn: sqlite3.Connection,
    knowledge_graph: nx.DiGraph,
    fragment_groups: list,
    min_cluster_size: int,
    snake_tokens: int,
) -> dict[tuple[int, int], tuple[list[list[int]], nx.DiGraph]]:
    """Detect snakes within each fragment group independently.

    Args:
        conn: Database connection
        knowledge_graph: Full knowledge graph
        fragment_groups: List of GroupInfo objects
        min_cluster_size: Minimum cluster size for snake detection
        snake_tokens: Maximum tokens per snake

    Returns:
        Dict mapping (chapter_id, group_id) to (snakes, snake_graph) tuples
    """
    group_snakes = {}

    for group in fragment_groups:
        print(f"\nProcessing Chapter {group.chapter_id}, Group {group.group_id}:")
        print(f"  Fragments: {group.fragment_ids}")

        # Get all chunks in this group's fragments
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(group.fragment_ids))
        cursor.execute(
            f"""
            SELECT id
            FROM chunks
            WHERE chapter_id = ? AND fragment_id IN ({placeholders})
        """,
            [group.chapter_id] + group.fragment_ids,
        )
        chunk_ids = [row[0] for row in cursor.fetchall()]

        if not chunk_ids:
            print("  No chunks in this group")
            group_snakes[(group.chapter_id, group.group_id)] = ([], nx.DiGraph())
            continue

        # Create subgraph with only chunks from this group
        group_graph = cast(nx.DiGraph, knowledge_graph.subgraph(chunk_ids).copy())

        # Add external edges to group_graph metadata (but don't use them for snake detection)
        external_edges = []
        for chunk_id in chunk_ids:
            for neighbor in knowledge_graph.neighbors(chunk_id):
                if neighbor not in chunk_ids:
                    external_edges.append((chunk_id, neighbor))
            for predecessor in knowledge_graph.predecessors(chunk_id):
                if predecessor not in chunk_ids:
                    external_edges.append((predecessor, chunk_id))

        print(f"  Group graph: {len(group_graph.nodes())} nodes, {len(group_graph.edges())} edges")
        print(f"  External edges: {len(external_edges)}")

        # Split into connected components within group
        components = split_connected_components(group_graph)
        print(f"  Found {len(components)} connected component(s)")

        # Detect snakes in each component
        detector = SnakeDetector(
            min_cluster_size=min_cluster_size,
            snake_tokens=snake_tokens,
        )

        all_snakes = []
        for i, component in enumerate(components):
            print(f"    Component {i}: {len(component.nodes())} nodes")
            component_snakes = detector.detect_snakes(component)
            all_snakes.extend(component_snakes)

        if not all_snakes:
            print("  No snakes detected")
            group_snakes[(group.chapter_id, group.group_id)] = ([], nx.DiGraph())
            continue

        print(f"  Found {len(all_snakes)} snakes")

        # Build snake graph for this group
        builder = SnakeGraphBuilder()
        snake_graph = builder.build_snake_graph(all_snakes, group_graph)

        # Add external edges to snake graph (inherited from chunks)
        for snake_id, snake in enumerate(all_snakes):
            for chunk_id in snake:
                for from_chunk, to_chunk in external_edges:
                    if from_chunk == chunk_id or to_chunk == chunk_id:
                        # This snake has external connections
                        if "external_edges" not in snake_graph.nodes[snake_id]:
                            snake_graph.nodes[snake_id]["external_edges"] = []
                        snake_graph.nodes[snake_id]["external_edges"].append((from_chunk, to_chunk))

        print(f"  Snake graph: {len(snake_graph.nodes())} snakes, {len(snake_graph.edges())} edges")

        group_snakes[(group.chapter_id, group.group_id)] = (all_snakes, snake_graph)

    return group_snakes


def _save_snakes_by_groups(
    conn: sqlite3.Connection,
    group_snakes: dict[tuple[int, int], tuple[list[list[int]], nx.DiGraph]],
    knowledge_graph: nx.DiGraph,
):
    """Save snakes to database with chapter and group information.

    Args:
        conn: Database connection
        group_snakes: Dict mapping (chapter_id, group_id) to (snakes, snake_graph)
        knowledge_graph: Knowledge graph (for node attributes)
    """
    cursor = conn.cursor()

    # Save snakes for each group
    for (chapter_id, group_id), (snakes, snake_graph) in group_snakes.items():
        for local_snake_id, snake in enumerate(snakes):
            first_node = knowledge_graph.nodes[snake[0]]
            last_node = knowledge_graph.nodes[snake[-1]]

            # Calculate total tokens and weight
            total_tokens = sum(knowledge_graph.nodes[chunk_id].get("tokens", 0) for chunk_id in snake)
            total_weight = sum(knowledge_graph.nodes[chunk_id].get("weight", 0.0) for chunk_id in snake)

            # Insert snake
            cursor.execute(
                """
                INSERT INTO snakes (chapter_id, group_id, local_snake_id, size, first_label, last_label, tokens, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    chapter_id,
                    group_id,
                    local_snake_id,
                    len(snake),
                    first_node["label"],
                    last_node["label"],
                    total_tokens,
                    total_weight,
                ),
            )

            # Get the auto-generated snake_id
            global_snake_id = cursor.lastrowid

            # Save snake-chunk associations
            for position, chunk_id in enumerate(snake):
                cursor.execute(
                    """
                    INSERT INTO snake_chunks (snake_id, chunk_id, position)
                    VALUES (?, ?, ?)
                """,
                    (global_snake_id, chunk_id, position),
                )

        # Save snake edges (need to map local snake IDs to global snake IDs)
        if snake_graph:
            # Build mapping from local_snake_id to global_snake_id
            cursor.execute(
                """
                SELECT id, local_snake_id
                FROM snakes
                WHERE chapter_id = ? AND group_id = ?
            """,
                (chapter_id, group_id),
            )
            id_mapping = {local_id: global_id for global_id, local_id in cursor.fetchall()}

            for from_local, to_local in snake_graph.edges():
                edge_data = snake_graph.edges[from_local, to_local]
                from_global = id_mapping[from_local]
                to_global = id_mapping[to_local]

                cursor.execute(
                    """
                    INSERT INTO snake_edges (from_snake_id, to_snake_id, weight)
                    VALUES (?, ?, ?)
                """,
                    (from_global, to_global, edge_data.get("weight", 0.1)),
                )

    conn.commit()


def _find_edge_strength(
    links: list[dict],
    from_id: int,
    to_id: int,
    chunks: list[CognitiveChunk],
    temp_ids: list[str],
) -> str | None:
    """Find the strength of an edge from the links data.

    Args:
        links: Raw link data from LLM (with from/to as temp_id or int)
        from_id: Actual from chunk ID (integer)
        to_id: Actual to chunk ID (integer)
        chunks: List of chunks with assigned IDs
        temp_ids: List of temp IDs corresponding to chunks

    Returns:
        Strength string or None if not found
    """
    # Build mapping from chunk ID to temp ID
    _id_to_temp = {chunk.id: temp_id for chunk, temp_id in zip(chunks, temp_ids)}

    # Try to find matching link
    for link in links:
        from_ref = link.get("from")
        to_ref = link.get("to")

        # Resolve from_ref to chunk ID
        from_chunk_id = None
        if isinstance(from_ref, int):
            from_chunk_id = from_ref
        elif isinstance(from_ref, str):
            # Find chunk with this temp_id
            for chunk, temp_id in zip(chunks, temp_ids):
                if temp_id == from_ref:
                    from_chunk_id = chunk.id
                    break

        # Resolve to_ref to chunk ID
        to_chunk_id = None
        if isinstance(to_ref, int):
            to_chunk_id = to_ref
        elif isinstance(to_ref, str):
            # Find chunk with this temp_id
            for chunk, temp_id in zip(chunks, temp_ids):
                if temp_id == to_ref:
                    to_chunk_id = chunk.id
                    break

        # Check if this link matches our edge
        if from_chunk_id == from_id and to_chunk_id == to_id:
            return link.get("strength")

    return None
