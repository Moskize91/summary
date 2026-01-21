import shutil
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
from tiktoken import Encoding

from ..llm import LLM
from . import database
from .api import Topologization
from .chunk_extraction import ChunkExtractor
from .cognitive_chunk import CognitiveChunk
from .fragment import FragmentWriter
from .fragment_grouping import group_fragments_by_chapter
from .graph_weights import add_weights_to_graph
from .snake_detector import SnakeDetector, split_connected_components
from .snake_graph_builder import SnakeGraphBuilder
from .text_fragmenter import TextFragmenter
from .wave_reflection import WaveReflection
from .working_memory import WorkingMemory


@dataclass
class TopologizationConfig:
    """Configuration for topologization pipeline."""

    # Processing parameters
    max_fragment_tokens: int = 800
    batch_size: int = 50000
    working_memory_capacity: int = 7
    generation_decay_factor: float = 0.5
    max_chunks: int | None = None  # None for unlimited

    # Snake detection parameters
    min_cluster_size: int = 2
    snake_tokens: int = 700

    # Fragment grouping parameters
    group_tokens_count: int = 10000  # Maximum tokens per fragment group


def topologize(
    intention: str,
    input: Iterable[Iterable[tuple[int, str]]],
    workspace_path: Path,
    config: TopologizationConfig,
    llm: LLM,
    encoding: Encoding,
) -> Topologization:
    """Execute topologization pipeline and create workspace.

    Args:
        intention: User's reading intention/goal
        input: Iterable of chapters, each chapter is an iterable of (token_count, sentence_text)
        workspace_path: Directory to store fragments + database
        config: Pipeline configuration
        llm: LLM instance for extraction and summarization
        encoding: Token encoding

    Returns:
        Topologization object for accessing results
    """
    print("=" * 60)
    print("=== Topologization Pipeline ===")
    print("=" * 60)
    print(f"Workspace: {workspace_path}")
    print(f"Working memory capacity: {config.working_memory_capacity}\n")

    # Step 1: Setup workspace
    if workspace_path.exists():
        shutil.rmtree(workspace_path)
    workspace_path.mkdir(parents=True)
    (workspace_path / "fragments").mkdir()

    # Step 2: Generate extraction guidance from intention
    print("\n=== Meta-Prompt: Generating Extraction Guidance ===")
    extraction_guidance = _generate_extraction_guidance(
        intention=intention,
        llm=llm,
    )

    # Step 3: Initialize components
    fragment_writer = FragmentWriter(workspace_path)
    fragmenter = TextFragmenter(fragment_writer, encoding, config.max_fragment_tokens)
    extractor = ChunkExtractor(llm, extraction_guidance)
    working_memory = WorkingMemory(capacity=config.working_memory_capacity)
    wave_reflection = WaveReflection(generation_decay_factor=config.generation_decay_factor)

    # Step 4: Extract knowledge graph
    print("\n=== Phase 1: Knowledge Graph Extraction ===")
    knowledge_graph, all_chunks = _extract_knowledge_graph(
        input,
        fragmenter,
        extractor,
        working_memory,
        wave_reflection,
        config.max_chunks,
    )

    # Finalize fragment writing
    fragmenter.finalize()

    print(f"\n{'=' * 60}")
    print("=== Knowledge Graph Results ===")
    print(f"{'=' * 60}")
    print(f"Total chunks: {knowledge_graph.number_of_nodes()}")
    print(f"Total connections: {knowledge_graph.number_of_edges()}")

    # Step 5: Initialize database
    db_path = workspace_path / "database.db"
    conn = database.initialize_database(db_path)

    # Step 6: Save chunks and edges to database
    print("\nSaving knowledge graph to database...")
    _save_knowledge_graph(conn, knowledge_graph, all_chunks)

    # Step 6.5: Group fragments by chapter using resource segmentation
    print(f"\n{'=' * 60}")
    print("=== Fragment Grouping ===")
    print(f"{'=' * 60}")
    print(f"Group token limit: {config.group_tokens_count}")

    fragment_groups = group_fragments_by_chapter(conn, workspace_path, config.group_tokens_count)
    print(f"\nCreated {len(fragment_groups)} fragment groups:")
    for group in fragment_groups:
        print(f"  Chapter {group.chapter_id}, Group {group.group_id}: {len(group.fragment_ids)} fragments")

    # Save fragment groups to database
    _save_fragment_groups(conn, fragment_groups)

    # Step 7: Detect and analyze snakes within each group
    print(f"\n{'=' * 60}")
    print("=== Phase 2: Thematic Chain Detection ===")
    print(f"{'=' * 60}")

    group_snakes = _analyze_snakes_by_groups(conn, knowledge_graph, fragment_groups, config)

    # Step 8: Save snakes to database (per group)
    print("\nSaving snakes to database...")
    _save_snakes_by_groups(conn, group_snakes, knowledge_graph)

    conn.close()

    # Calculate total snakes across all groups
    total_snakes = sum(len(snakes_list) for snakes_list, _ in group_snakes.values())

    print(f"\n{'=' * 60}")
    print("=== Pipeline Complete ===")
    print(f"{'=' * 60}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Total snakes: {total_snakes}")
    print(f"Workspace: {workspace_path}")

    # Step 9: Return Topologization object
    return Topologization(workspace_path)


def _generate_extraction_guidance(
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
    response = llm.request(
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


def _extract_knowledge_graph(
    input: Iterable[Iterable[tuple[int, str]]],
    fragmenter: TextFragmenter,
    extractor: ChunkExtractor,
    working_memory: WorkingMemory,
    wave_reflection: WaveReflection,
    max_chunks: int | None,
) -> tuple[nx.DiGraph, list[CognitiveChunk]]:
    """Extract knowledge graph from input with two-stage extraction.

    Args:
        input: Iterable of chapters, each chapter is an iterable of (token_count, sentence_text)
        fragmenter: Text fragmenter
        extractor: Chunk extractor
        working_memory: Working memory
        wave_reflection: Wave reflection
        max_chunks: Maximum chunks to process (None for unlimited)

    Returns:
        Tuple of (knowledge_graph, all_chunks)
    """
    knowledge_graph = nx.DiGraph()
    all_chunks: list[CognitiveChunk] = []
    chunk_count = 0

    for fragment_with_sentences in fragmenter.stream_fragments(input):
        chunk_count += 1

        if max_chunks is not None and chunk_count > max_chunks:
            print(f"Reached max chunks limit ({max_chunks}), stopping...")
            break

        print(f"Processing fragment {chunk_count}...")

        # === Stage 1: Extract user-focused chunks ===
        user_focused_result, fragment_summary = extractor.extract_user_focused(
            fragment_with_sentences.text,
            working_memory,
            fragment_with_sentences.sentence_ids,
            fragment_with_sentences.sentence_texts,
            fragment_with_sentences.sentence_token_counts,
        )

        if user_focused_result is None:
            print(f"Warning: User-focused extraction failed for fragment {chunk_count}")
            continue

        # Store fragment summary (will be written when fragment is ended)
        if fragment_summary:
            fragmenter.fragment_writer.set_summary(fragment_summary)

        # Add user-focused chunks to working memory and assign IDs
        user_focused_chunks, user_focused_edges = working_memory.add_chunks_with_links(user_focused_result)

        # Add user-focused chunks to knowledge graph
        for chunk in user_focused_chunks:
            knowledge_graph.add_node(
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
            knowledge_graph.add_edge(from_id, to_id, strength=strength)

        all_chunks.extend(user_focused_chunks)

        # === Stage 2: Extract book-coherence chunks ===
        book_coherence_result = extractor.extract_book_coherence(
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
            book_coherence_chunks, book_coherence_edges = working_memory.add_chunks_with_links(book_coherence_result)

            # Add book-coherence chunks to knowledge graph
            for chunk in book_coherence_chunks:
                knowledge_graph.add_node(
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
            # Note: Only pass book_coherence chunks/temp_ids, since user_focused chunks
            # are already referenced by integer IDs in the links
            for from_id, to_id in book_coherence_edges:
                strength = _find_edge_strength(
                    book_coherence_result.links, from_id, to_id, book_coherence_chunks, book_coherence_result.temp_ids
                )
                knowledge_graph.add_edge(from_id, to_id, strength=strength)

            all_chunks.extend(book_coherence_chunks)

        # === Update working memory with wave reflection ===
        # Get all chunks from current fragment (both stages)
        current_fragment_chunk_ids = [c.id for c in working_memory.get_all_chunks_for_saving()]

        # Select extra chunks from history using wave reflection
        extra_chunks = wave_reflection.select_top_chunks(
            all_chunks=all_chunks,
            knowledge_graph=knowledge_graph,
            latest_chunk_ids=current_fragment_chunk_ids,
            capacity=working_memory.capacity,
        )

        # Set extra chunks and finalize fragment
        working_memory.set_extra_chunks(extra_chunks)
        working_memory.finalize_fragment()

    # Compute and add weights to knowledge graph
    print("\nComputing node and edge weights...")
    add_weights_to_graph(knowledge_graph)

    return knowledge_graph, all_chunks


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
    config: TopologizationConfig,
) -> dict[tuple[int, int], tuple[list[list[int]], nx.DiGraph]]:
    """Detect snakes within each fragment group independently.

    Args:
        conn: Database connection
        knowledge_graph: Full knowledge graph
        fragment_groups: List of GroupInfo objects
        config: Pipeline configuration

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
            min_cluster_size=config.min_cluster_size,
            snake_tokens=config.snake_tokens,
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


def _analyze_snakes(knowledge_graph: nx.DiGraph, config: TopologizationConfig) -> tuple[list[list[int]], nx.DiGraph]:
    # Split into connected components
    print("\nSplitting into connected components...")
    components = split_connected_components(knowledge_graph)
    print(f"Found {len(components)} connected component(s):")
    for i, comp in enumerate(components):
        print(f"  Component {i}: {len(comp.nodes())} nodes, {len(comp.edges())} edges")

    # Detect snakes in each component
    print("\nDetecting thematic chains (snakes)...")
    detector = SnakeDetector(
        min_cluster_size=config.min_cluster_size,
        snake_tokens=config.snake_tokens,
    )

    all_snakes = []
    for i, component in enumerate(components):
        print(f"\nProcessing Component {i}:")
        component_snakes = detector.detect_snakes(component)
        all_snakes.extend(component_snakes)

    if not all_snakes:
        print("No snakes detected")
        return [], nx.DiGraph()

    print(f"\nFound {len(all_snakes)} snakes")

    # Build snake graph
    print(f"\n{'=' * 60}")
    print("=== Building Snake Graph ===")
    print(f"{'=' * 60}")

    builder = SnakeGraphBuilder()
    snake_graph = builder.build_snake_graph(all_snakes, knowledge_graph)

    print(f"Snake graph: {len(snake_graph.nodes())} snakes, {len(snake_graph.edges())} inter-snake edges")

    return all_snakes, snake_graph


def _save_snakes(
    conn: sqlite3.Connection,
    snakes: list[list[int]],
    snake_graph: nx.DiGraph,
    knowledge_graph: nx.DiGraph,
):
    """Save snakes to database.

    Args:
        conn: Database connection
        snakes: List of snakes
        snake_graph: Snake graph
        knowledge_graph: Knowledge graph (for node attributes)
    """
    # Save snakes
    for snake_id, snake in enumerate(snakes):
        first_node = knowledge_graph.nodes[snake[0]]
        last_node = knowledge_graph.nodes[snake[-1]]

        # Calculate total tokens and weight for this snake
        total_tokens = sum(knowledge_graph.nodes[chunk_id].get("tokens", 0) for chunk_id in snake)
        total_weight = sum(knowledge_graph.nodes[chunk_id].get("weight", 0.0) for chunk_id in snake)

        database.insert_snake(
            conn,
            snake_id,
            len(snake),
            first_node["label"],
            last_node["label"],
            tokens=total_tokens,
            weight=total_weight,
        )

        # Save snake-chunk associations
        for position, chunk_id in enumerate(snake):
            database.insert_snake_chunk(conn, snake_id, chunk_id, position)

    # Save snake edges
    for from_snake, to_snake in snake_graph.edges():
        edge_data = snake_graph.edges[from_snake, to_snake]
        database.insert_snake_edge(
            conn,
            from_snake,
            to_snake,
            edge_data["weight"],
        )


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
