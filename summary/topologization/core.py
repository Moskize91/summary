"""Core API for topologization module - unified interface for cognitive chunk extraction."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from jinja2 import Environment

from ..llm import LLM
from .cognitive_chunk import CognitiveChunk
from .extractor import ChunkExtractor
from .snake_detector import SnakeDetector, split_connected_components
from .snake_graph_builder import SnakeGraphBuilder
from .snake_summarizer import SnakeSummarizer
from .text_chunker import TextChunker
from .wave_reflection import WaveReflection
from .working_memory import WorkingMemory


@dataclass
class PipelineConfig:
    """Configuration for TopologizationPipeline."""

    # Input/Output
    input_file: Path
    output_dir: Path

    # Prompt templates
    extraction_prompt_file: Path
    snake_summary_prompt_file: Path

    # Processing parameters
    max_chunk_length: int = 800
    batch_size: int = 50000
    working_memory_capacity: int = 7
    generation_decay_factor: float = 0.68
    max_chunks: int | None = 40  # None for unlimited

    # Snake detection parameters
    min_cluster_size: int = 2
    phase2_stop_ratio: float = 0.15

    # Output control
    clear_output_on_start: bool = True
    save_intermediate_results: bool = True


@dataclass
class PipelineResult:
    """Result from TopologizationPipeline execution."""

    # Knowledge graph
    knowledge_graph: nx.DiGraph
    all_chunks: list[CognitiveChunk]

    # Snakes
    snakes: list[list[int]]
    snake_summaries: list[dict]
    snake_graph: nx.DiGraph

    # Output files
    output_files: dict[str, Path]


class TopologizationPipeline:
    """Complete pipeline for cognitive chunk extraction and thematic chain detection.

    This is the high-level API that combines all steps:
    1. Text chunking and sentence detection
    2. Cognitive chunk extraction with LLM
    3. Working memory management with Wave Reflection
    4. Knowledge graph construction
    5. Snake (thematic chain) detection
    6. Snake summarization
    7. Snake graph construction
    """

    def __init__(self, config: PipelineConfig, llm: LLM, jinja_env: Environment):
        """Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
            llm: LLM instance for extraction and summarization
            jinja_env: Jinja2 Environment for loading prompt templates
        """
        self.config = config
        self.llm = llm
        self.jinja_env = jinja_env

        # Setup directories
        self._setup_directories()

        # Initialize components
        self.chunker = TextChunker(
            max_chunk_length=config.max_chunk_length,
            batch_size=config.batch_size,
        )

        self.extractor = ChunkExtractor(self.llm, config.extraction_prompt_file, self.jinja_env)
        self.working_memory = WorkingMemory(capacity=config.working_memory_capacity)
        self.wave_reflection = WaveReflection(generation_decay_factor=config.generation_decay_factor)
        self.snake_detector = SnakeDetector(
            min_cluster_size=config.min_cluster_size,
            phase2_stop_ratio=config.phase2_stop_ratio,
        )
        self.snake_summarizer = SnakeSummarizer(self.llm, config.snake_summary_prompt_file, self.jinja_env)

    def _setup_directories(self) -> None:
        """Setup output directories."""
        if self.config.clear_output_on_start and self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> PipelineResult:
        """Run the complete pipeline.

        Returns:
            PipelineResult containing all outputs
        """
        print("=== Cognitive Chunk Extraction ===")
        print(f"Working memory capacity: {self.config.working_memory_capacity}")
        print(f"Input file: {self.config.input_file.name}")
        print(f"Output directory: {self.config.output_dir}\n")

        # Step 1: Extract knowledge graph
        print("\n=== Phase 1: Knowledge Graph Extraction ===")
        knowledge_graph, all_chunks = self._extract_knowledge_graph()

        # Step 2: Detect and analyze snakes
        print(f"\n{'=' * 60}")
        print("=== Phase 2: Thematic Chain Detection ===")
        print(f"{'=' * 60}")
        snakes, snake_summaries, snake_graph = self._analyze_snakes(knowledge_graph)

        # Save results
        output_files = {}
        if self.config.save_intermediate_results:
            output_files = self._save_results(knowledge_graph, snakes, snake_summaries, snake_graph)

        return PipelineResult(
            knowledge_graph=knowledge_graph,
            all_chunks=all_chunks,
            snakes=snakes,
            snake_summaries=snake_summaries,
            snake_graph=snake_graph,
            output_files=output_files,
        )

    def _extract_knowledge_graph(self) -> tuple[nx.DiGraph, list[CognitiveChunk]]:
        """Extract knowledge graph from text.

        Returns:
            Tuple of (knowledge_graph, all_chunks)
        """
        knowledge_graph = nx.DiGraph()
        all_chunks = []
        chunk_count = 0
        max_chunks = self.config.max_chunks

        for i, chunk_with_sentences in enumerate(self.chunker.stream_chunks_from_file(self.config.input_file)):
            if max_chunks is not None and i >= max_chunks:
                break

            chunk_count += 1
            text_segment = chunk_with_sentences.text
            sentence_map = self.chunker.get_sentence_map()

            print(f"\n{'=' * 60}")
            print(f"Processing segment {chunk_count}")
            print(f"{'=' * 60}")
            print(f"Text preview: {text_segment[:100]}...")
            print(f"Sentence IDs: {chunk_with_sentences.sentence_ids[:5]}...")
            print(f"\nCurrent working memory ({len(self.working_memory.get_chunks())} chunks):")
            print(self.working_memory.format_for_prompt())

            # Extract chunks
            print("\nExtracting cognitive chunks...")
            extraction_result = self.extractor.extract_chunks(text_segment, self.working_memory, sentence_map)

            if extraction_result is None:
                print("\nNo chunks extracted (LLM failed or returned empty)")
                continue

            # Check if JSON key order was correct
            if not extraction_result.order_correct:
                print("\n⚠️  WARNING: JSON key order was incorrect (links before chunks)")
                print("   The LLM should output 'chunks' first, then 'links'")

            if extraction_result.chunks:
                print(f"\nExtracted {len(extraction_result.chunks)} new chunks:")
                for chunk, temp_id in zip(extraction_result.chunks, extraction_result.temp_ids):
                    print(f"  - (temp_id: {temp_id}) [{chunk.label}] {chunk.content[:80]}...")

                # Process links and add to working memory
                added_chunks, edges = self.working_memory.add_chunks_with_links(extraction_result)

                # Add chunks to all_chunks list
                all_chunks.extend(added_chunks)

                # Track latest chunk IDs for Wave Reflection
                latest_chunk_ids = [chunk.id for chunk in added_chunks]

                # Add nodes to knowledge graph
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

                print(f"\nProcessed {len(edges)} links:")
                for from_id, to_id in edges:
                    print(f"  - {from_id} -> {to_id}")

                print(f"\nAdded {len(added_chunks)} chunks to knowledge graph")

                # Use Wave Reflection to select top chunks for working memory
                if all_chunks:
                    selected_chunks = self.wave_reflection.select_top_chunks(
                        all_chunks=all_chunks,
                        knowledge_graph=knowledge_graph,
                        latest_chunk_ids=latest_chunk_ids,
                        capacity=self.working_memory.capacity,
                    )
                    self.working_memory._chunks = selected_chunks
                    print(f"\nWave Reflection selected {len(selected_chunks)} chunks for working memory")
            else:
                print("\nNo chunks extracted from this segment")

        print(f"\n\n{'=' * 60}")
        print("=== Knowledge Graph Results ===")
        print(f"{'=' * 60}")
        print(f"Total text segments processed: {chunk_count}")
        print(f"Total chunks in knowledge graph: {knowledge_graph.number_of_nodes()}")
        print(f"Total connections: {knowledge_graph.number_of_edges()}")

        return knowledge_graph, all_chunks

    def _analyze_snakes(self, knowledge_graph: nx.DiGraph) -> tuple[list[list[int]], list[dict], nx.DiGraph]:
        """Detect snakes and generate summaries.

        Args:
            knowledge_graph: Knowledge graph with chunks and edges

        Returns:
            Tuple of (snakes, snake_summaries, snake_graph)
        """
        # Split into connected components
        print("Splitting into connected components...")
        components = split_connected_components(knowledge_graph)
        print(f"Found {len(components)} connected component(s):")
        for i, comp in enumerate(components):
            print(f"  Component {i}: {len(comp.nodes())} nodes, {len(comp.edges())} edges")

        # Detect snakes
        print("\nDetecting thematic chains (snakes)...")
        all_snakes = []
        for i, component in enumerate(components):
            print(f"\nProcessing Component {i}:")
            component_snakes = self.snake_detector.detect_snakes(component)
            all_snakes.extend(component_snakes)

        if not all_snakes:
            print("\nNo snakes detected")
            return [], [], nx.DiGraph()

        print(f"\nFound {len(all_snakes)} snakes:")
        for i, snake in enumerate(all_snakes):
            first_node = knowledge_graph.nodes[snake[0]]
            last_node = knowledge_graph.nodes[snake[-1]]
            print(f"  Snake {i}: {len(snake)} nodes - {first_node['label']} → {last_node['label']}")

        # Summarize snakes
        print(f"\n{'=' * 60}")
        print("=== Snake Summarization ===")
        print(f"{'=' * 60}")
        print(f"Generating summaries for {len(all_snakes)} snakes...")
        snake_summaries = self.snake_summarizer.summarize_all_snakes(all_snakes, knowledge_graph)

        print("\nSample summaries:")
        for summary in snake_summaries[:3]:
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

    def _save_results(
        self,
        knowledge_graph: nx.DiGraph,
        snakes: list[list[int]],
        snake_summaries: list[dict],
        snake_graph: nx.DiGraph,
    ) -> dict[str, Path]:
        """Save all results to files.

        Returns:
            Dictionary mapping result names to file paths
        """
        output_files = {}

        # Save knowledge graph
        json_output_file = self.config.output_dir / "knowledge_graph.json"
        graph_data = {
            "nodes": [
                {
                    "id": n,
                    "generation": knowledge_graph.nodes[n]["generation"],
                    "sentence_id": knowledge_graph.nodes[n]["sentence_id"],
                    "label": knowledge_graph.nodes[n]["label"],
                    "content": knowledge_graph.nodes[n]["content"],
                }
                for n in knowledge_graph.nodes()
            ],
            "edges": [{"from": u, "to": v} for u, v in knowledge_graph.edges()],
        }

        with open(json_output_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        output_files["knowledge_graph"] = json_output_file
        print(f"\nKnowledge graph saved to: {json_output_file}")
        print(f"  - {len(graph_data['nodes'])} nodes")
        print(f"  - {len(graph_data['edges'])} edges")

        # Save snakes
        snakes_output = self.config.output_dir / "snakes.json"
        snakes_data = []
        for i, snake in enumerate(snakes):
            snake_nodes = []
            for node_id in snake:
                node_data = knowledge_graph.nodes[node_id]
                snake_nodes.append(
                    {
                        "id": node_id,
                        "sentence_id": node_data["sentence_id"],
                        "label": node_data["label"],
                    }
                )
            snakes_data.append({"snake_id": i, "size": len(snake), "nodes": snake_nodes})

        with open(snakes_output, "w", encoding="utf-8") as f:
            json.dump(snakes_data, f, ensure_ascii=False, indent=2)

        output_files["snakes"] = snakes_output
        print(f"\nSnakes saved to: {snakes_output}")

        # Save summaries
        summaries_output = self.config.output_dir / "snake_summaries.json"
        with open(summaries_output, "w", encoding="utf-8") as f:
            json.dump(snake_summaries, f, ensure_ascii=False, indent=2)

        output_files["snake_summaries"] = summaries_output
        print(f"\nSnake summaries saved to: {summaries_output}")

        # Save snake edges
        edges_output = self.config.output_dir / "snake_edges.json"
        edges_data = []
        for edge in snake_graph.edges():
            snake_from, snake_to = edge
            edge_data = snake_graph.edges[edge]

            from_node = knowledge_graph.nodes[snakes[snake_from][0]]
            to_node = knowledge_graph.nodes[snakes[snake_to][0]]

            edges_data.append(
                {
                    "from_snake": snake_from,
                    "to_snake": snake_to,
                    "from_label": from_node["label"],
                    "to_label": to_node["label"],
                    "internal_edge_count": edge_data["internal_edge_count"],
                }
            )

        with open(edges_output, "w", encoding="utf-8") as f:
            json.dump(edges_data, f, ensure_ascii=False, indent=2)

        output_files["snake_edges"] = edges_output
        print(f"\nSnake edges saved to: {edges_output}")

        return output_files


class KnowledgeGraphExtractor:
    """Mid-level API for knowledge graph extraction only.

    Use this when you only need the knowledge graph without snake detection.
    """

    def __init__(
        self,
        llm: LLM,
        extraction_prompt_file: Path,
        jinja_env: Environment,
        max_chunk_length: int = 800,
        batch_size: int = 50000,
        working_memory_capacity: int = 7,
        generation_decay_factor: float = 0.68,
    ):
        """Initialize knowledge graph extractor.

        Args:
            llm: LLM instance for extraction
            extraction_prompt_file: Path to extraction prompt template
            jinja_env: Jinja2 Environment for loading templates
            max_chunk_length: Maximum character length per chunk
            batch_size: Text batch size for spacy processing
            working_memory_capacity: Working memory capacity
            generation_decay_factor: Decay factor for Wave Reflection
        """
        self.chunker = TextChunker(max_chunk_length=max_chunk_length, batch_size=batch_size)
        self.extractor = ChunkExtractor(llm, extraction_prompt_file, jinja_env)
        self.working_memory = WorkingMemory(capacity=working_memory_capacity)
        self.wave_reflection = WaveReflection(generation_decay_factor=generation_decay_factor)

    def extract_from_file(
        self, input_file: Path, max_chunks: int | None = None
    ) -> tuple[nx.DiGraph, list[CognitiveChunk]]:
        """Extract knowledge graph from a text file.

        Args:
            input_file: Path to input text file
            max_chunks: Maximum number of text chunks to process (None for unlimited)

        Returns:
            Tuple of (knowledge_graph, all_chunks)
        """
        knowledge_graph = nx.DiGraph()
        all_chunks = []

        for i, chunk_with_sentences in enumerate(self.chunker.stream_chunks_from_file(input_file)):
            if max_chunks is not None and i >= max_chunks:
                break

            text_segment = chunk_with_sentences.text
            sentence_map = self.chunker.get_sentence_map()

            extraction_result = self.extractor.extract_chunks(text_segment, self.working_memory, sentence_map)

            if extraction_result is None or not extraction_result.chunks:
                continue

            # Process chunks and links
            added_chunks, edges = self.working_memory.add_chunks_with_links(extraction_result)
            all_chunks.extend(added_chunks)
            latest_chunk_ids = [chunk.id for chunk in added_chunks]

            # Add to knowledge graph
            for chunk in added_chunks:
                knowledge_graph.add_node(
                    chunk.id,
                    generation=chunk.generation,
                    sentence_id=chunk.sentence_id,
                    label=chunk.label,
                    content=chunk.content,
                )

            for from_id, to_id in edges:
                knowledge_graph.add_edge(from_id, to_id)

            # Update working memory with Wave Reflection
            if all_chunks:
                selected_chunks = self.wave_reflection.select_top_chunks(
                    all_chunks=all_chunks,
                    knowledge_graph=knowledge_graph,
                    latest_chunk_ids=latest_chunk_ids,
                    capacity=self.working_memory.capacity,
                )
                self.working_memory._chunks = selected_chunks

        return knowledge_graph, all_chunks


class ThematicChainAnalyzer:
    """Mid-level API for thematic chain (snake) detection and analysis.

    Use this when you already have a knowledge graph and want to detect snakes.
    """

    def __init__(
        self,
        min_cluster_size: int = 2,
        phase2_stop_ratio: float = 0.15,
    ):
        """Initialize thematic chain analyzer.

        Args:
            min_cluster_size: Minimum nodes in a snake
            phase2_stop_ratio: Phase 2 stop ratio for snake detection
        """
        self.snake_detector = SnakeDetector(
            min_cluster_size=min_cluster_size,
            phase2_stop_ratio=phase2_stop_ratio,
        )

    def detect_and_summarize(
        self,
        knowledge_graph: nx.DiGraph,
        llm: LLM,
        snake_summary_prompt_file: Path,
        jinja_env: Environment,
    ) -> tuple[list[list[int]], list[dict], nx.DiGraph]:
        """Detect snakes and generate summaries.

        Args:
            knowledge_graph: Knowledge graph with chunks and edges
            llm: LLM instance for summarization
            snake_summary_prompt_file: Path to snake summary prompt template
            jinja_env: Jinja2 Environment for loading templates

        Returns:
            Tuple of (snakes, snake_summaries, snake_graph)
        """
        # Split into components and detect snakes
        components = split_connected_components(knowledge_graph)
        all_snakes = []
        for component in components:
            component_snakes = self.snake_detector.detect_snakes(component)
            all_snakes.extend(component_snakes)

        if not all_snakes:
            return [], [], nx.DiGraph()

        # Summarize snakes
        summarizer = SnakeSummarizer(llm, snake_summary_prompt_file, jinja_env)
        snake_summaries = summarizer.summarize_all_snakes(all_snakes, knowledge_graph)

        # Build snake graph
        builder = SnakeGraphBuilder()
        snake_graph = builder.build_snake_graph(all_snakes, knowledge_graph)

        return all_snakes, snake_summaries, snake_graph
