import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from ..llm import LLM
from . import database
from .api import Topologization
from .chunk_extraction import ChunkExtractor
from .cognitive_chunk import CognitiveChunk
from .snake_detector import SnakeDetector, split_connected_components
from .snake_graph_builder import SnakeGraphBuilder
from .snake_summarizer import SnakeSummarizer
from .storage import FragmentWriter
from .text_chunker import TextChunker
from .wave_reflection import WaveReflection
from .working_memory import WorkingMemory


@dataclass
class TopologizationConfig:
    """Configuration for topologization pipeline."""

    # Processing parameters
    max_chunk_length: int = 800
    batch_size: int = 50000
    working_memory_capacity: int = 7
    generation_decay_factor: float = 0.68
    max_chunks: int | None = None  # None for unlimited

    # Snake detection parameters
    min_cluster_size: int = 2
    phase2_stop_ratio: float = 0.15


def topologize(
    intention: str,
    input_file: Path,
    workspace_path: Path,
    config: TopologizationConfig,
    llm: LLM,
) -> Topologization:
    """Execute topologization pipeline and create workspace.

    Args:
        input_file: Input text file to process
        workspace_path: Directory to store fragments + database
        config: Pipeline configuration
        llm: LLM instance for extraction and summarization

    Returns:
        Topologization object for accessing results
    """
    print("=" * 60)
    print("=== Topologization Pipeline ===")
    print("=" * 60)
    print(f"Input: {input_file}")
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
    chunker = TextChunker(fragment_writer, config.max_chunk_length, config.batch_size)
    extractor = ChunkExtractor(llm, extraction_guidance)
    working_memory = WorkingMemory(capacity=config.working_memory_capacity)
    wave_reflection = WaveReflection(generation_decay_factor=config.generation_decay_factor)

    # Step 4: Extract knowledge graph
    print("\n=== Phase 1: Knowledge Graph Extraction ===")
    knowledge_graph, all_chunks = _extract_knowledge_graph(
        input_file,
        chunker,
        extractor,
        working_memory,
        wave_reflection,
        config.max_chunks,
    )

    # Finalize fragment writing
    chunker.finalize()

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

    # Step 7: Detect and analyze snakes
    print(f"\n{'=' * 60}")
    print("=== Phase 2: Thematic Chain Detection ===")
    print(f"{'=' * 60}")

    snakes, snake_summaries, snake_graph = _analyze_snakes(
        knowledge_graph,
        all_chunks,
        config,
        llm,
    )

    # Step 8: Save snakes to database
    print("\nSaving snakes to database...")
    _save_snakes(conn, snakes, snake_summaries, snake_graph, knowledge_graph)

    conn.close()

    print(f"\n{'=' * 60}")
    print("=== Pipeline Complete ===")
    print(f"{'=' * 60}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Total snakes: {len(snakes)}")
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
    intention_prompt_file = Path(__file__).parent.parent / "data" / "intention" / "chunk_extraction.jinja"
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
    input_file: Path,
    chunker: TextChunker,
    extractor: ChunkExtractor,
    working_memory: WorkingMemory,
    wave_reflection: WaveReflection,
    max_chunks: int | None,
) -> tuple[nx.DiGraph, list[CognitiveChunk]]:
    """Extract knowledge graph from input file.

    Args:
        input_file: Input text file
        chunker: Text chunker
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

    for chunk_with_sentences in chunker.stream_chunks_from_file(input_file):
        chunk_count += 1

        if max_chunks is not None and chunk_count > max_chunks:
            print(f"Reached max chunks limit ({max_chunks}), stopping...")
            break

        print(f"Processing chunk {chunk_count}...")

        # Extract cognitive chunks from text
        extraction_result = extractor.extract_chunks(
            chunk_with_sentences.text,
            working_memory,
            chunk_with_sentences.sentence_ids,
            chunk_with_sentences.sentence_texts,
        )

        if extraction_result is None:
            print(f"Warning: Extraction failed for chunk {chunk_count}")
            continue

        # Add chunks with links to working memory
        added_chunks, edges = working_memory.add_chunks_with_links(extraction_result)

        # Add chunks to knowledge graph
        for chunk in added_chunks:
            knowledge_graph.add_node(
                chunk.id,
                generation=chunk.generation,
                sentence_id=chunk.sentence_id,
                label=chunk.label,
                content=chunk.content,
            )

        # Add edges to knowledge graph
        for from_id, to_id in edges:
            knowledge_graph.add_edge(from_id, to_id)

        # Collect all chunks
        all_chunks.extend(added_chunks)

        # Update working memory with wave reflection
        latest_chunk_ids = [chunk.id for chunk in added_chunks]
        selected_chunks = wave_reflection.select_top_chunks(
            working_memory.get_chunks(),
            knowledge_graph,
            latest_chunk_ids,
            capacity=working_memory.capacity,
        )
        working_memory._chunks = selected_chunks  # pylint: disable=protected-access

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
    # Save chunks
    for chunk in all_chunks:
        # Use the matched sentence IDs from source_sentences
        sentence_ids = chunk.sentence_ids if chunk.sentence_ids else [chunk.sentence_id]

        database.insert_chunk(
            conn,
            chunk.id,
            chunk.generation,
            chunk.sentence_id,
            chunk.label,
            sentence_ids,
        )

    # Save edges
    for from_id, to_id in knowledge_graph.edges():
        database.insert_edge(conn, from_id, to_id)


def _analyze_snakes(
    knowledge_graph: nx.DiGraph,
    _all_chunks: list[CognitiveChunk],
    config: TopologizationConfig,
    llm: LLM,
) -> tuple[list[list[int]], list[dict], nx.DiGraph]:
    """Detect snakes and generate summaries.

    Args:
        knowledge_graph: Knowledge graph
        _all_chunks: All chunks (unused, reserved for future use)
        config: Pipeline configuration
        llm: LLM instance

    Returns:
        Tuple of (snakes, snake_summaries, snake_graph)
    """
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
        phase2_stop_ratio=config.phase2_stop_ratio,
    )

    all_snakes = []
    for i, component in enumerate(components):
        print(f"\nProcessing Component {i}:")
        component_snakes = detector.detect_snakes(component)
        all_snakes.extend(component_snakes)

    if not all_snakes:
        print("No snakes detected")
        return [], [], nx.DiGraph()

    print(f"\nFound {len(all_snakes)} snakes")

    # Summarize snakes
    print("\nGenerating snake summaries...")
    summarizer = SnakeSummarizer(llm)
    snake_summaries = summarizer.summarize_all_snakes(all_snakes, knowledge_graph)

    for summary in snake_summaries:
        print(f"\n  Snake {summary['snake_id']}: {summary['first_label']} → {summary['last_label']}")
        print(f"  Summary: {summary['summary'][:100]}...")

    # Build snake graph
    print(f"\n{'=' * 60}")
    print("=== Building Snake Graph ===")
    print(f"{'=' * 60}")

    builder = SnakeGraphBuilder()
    snake_graph = builder.build_snake_graph(all_snakes, knowledge_graph)

    print(f"Snake graph: {len(snake_graph.nodes())} snakes, {len(snake_graph.edges())} inter-snake edges")

    return all_snakes, snake_summaries, snake_graph


def _save_snakes(
    conn: sqlite3.Connection,
    snakes: list[list[int]],
    snake_summaries: list[dict],
    snake_graph: nx.DiGraph,
    knowledge_graph: nx.DiGraph,
):
    """Save snakes to database.

    Args:
        conn: Database connection
        snakes: List of snakes
        snake_summaries: Snake summaries
        snake_graph: Snake graph
        knowledge_graph: Knowledge graph (for node attributes)
    """
    # Save snakes
    for snake_id, snake in enumerate(snakes):
        first_node = knowledge_graph.nodes[snake[0]]
        last_node = knowledge_graph.nodes[snake[-1]]

        database.insert_snake(
            conn,
            snake_id,
            len(snake),
            first_node["label"],
            last_node["label"],
        )

        # Save snake-chunk associations
        for position, chunk_id in enumerate(snake):
            database.insert_snake_chunk(conn, snake_id, chunk_id, position)

    # Save snake summaries
    for summary in snake_summaries:
        database.insert_snake_summary(
            conn,
            summary["snake_id"],
            summary["summary"],
        )

    # Save snake edges
    for from_snake, to_snake in snake_graph.edges():
        edge_data = snake_graph.edges[from_snake, to_snake]
        database.insert_snake_edge(
            conn,
            from_snake,
            to_snake,
            edge_data["internal_edge_count"],
        )
