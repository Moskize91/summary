"""Main entry point for cognitive chunk extraction."""

import json
import shutil
from pathlib import Path

import networkx as nx

from .extractor import ChunkExtractor
from .llm import LLM
from .snake_detector import SnakeDetector, split_connected_components
from .snake_summarizer import SnakeSummarizer
from .text_chunker import TextChunker
from .wave_reflection import WaveReflection
from .working_memory import WorkingMemory


def main():
    """Main function to run the cognitive chunk extraction process."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = Path(__file__).parent / "data"
    input_file = data_dir / "明朝那些事儿.txt"
    config_file = project_root / "format.json"
    prompt_file = data_dir / "extraction_prompt.jinja"

    # Output directory structure
    output_dir = project_root / "output"
    log_dir = output_dir / "logs"
    cache_dir = project_root / "cache"  # Cache outside output for persistence

    # Clear output directory on each run (but keep cache)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    chunker = TextChunker(max_chunk_length=800, batch_size=50000)
    llm = LLM(config_path=config_file, log_dir_path=log_dir, cache_dir_path=cache_dir)
    extractor = ChunkExtractor(llm, prompt_file)
    working_memory = WorkingMemory(capacity=7)
    wave_reflection = WaveReflection(generation_decay_factor=0.68)

    # Initialize graph to store all chunks
    knowledge_graph = nx.DiGraph()

    # Track all chunks for Wave Reflection
    all_chunks = []

    print("=== Cognitive Chunk Extraction ===")
    print(f"Working memory capacity: {working_memory.capacity}")
    print(f"Input file: {input_file.name}")
    print(f"Output directory: {output_dir}")
    print(f"  - Logs: {log_dir}")
    print(f"Cache directory: {cache_dir} (persistent)\n")

    # Process text chunks
    chunk_count = 0
    max_chunks = 40  # Process first 40 chunks for testing

    for i, chunk_with_sentences in enumerate(chunker.stream_chunks_from_file(input_file)):
        if i >= max_chunks:
            break

        chunk_count += 1
        text_segment = chunk_with_sentences.text
        sentence_map = chunker.get_sentence_map()

        print(f"\n{'=' * 60}")
        print(f"Processing segment {chunk_count}")
        print(f"{'=' * 60}")
        print(f"Text preview: {text_segment[:100]}...")
        print(f"Sentence IDs: {chunk_with_sentences.sentence_ids[:5]}...")
        print(f"\nCurrent working memory ({len(working_memory.get_chunks())} chunks):")
        print(working_memory.format_for_prompt())

        # Extract chunks
        print("\nExtracting cognitive chunks...")
        extraction_result = extractor.extract_chunks(text_segment, working_memory, sentence_map)

        if extraction_result is None:
            print("\nNo chunks extracted (LLM failed or returned empty)")
            continue

        # Check if JSON key order was correct
        if not extraction_result.order_correct:
            print("\n⚠️  WARNING: JSON key order was incorrect (links before chunks)")
            print("   The LLM should output 'chunks' first, then 'links'")
            print("   This extraction will still be processed, but consider retrying")
            print("   if this happens frequently.")

        if extraction_result.chunks:
            print(f"\nExtracted {len(extraction_result.chunks)} new chunks:")
            for chunk, temp_id in zip(extraction_result.chunks, extraction_result.temp_ids):
                print(f"  - (temp_id: {temp_id}) [{chunk.label}] {chunk.content[:80]}...")

            # Process links and add to working memory
            added_chunks, edges = working_memory.add_chunks_with_links(extraction_result)

            # Add chunks to all_chunks list
            all_chunks.extend(added_chunks)

            # Track latest chunk IDs for Wave Reflection
            latest_chunk_ids = [chunk.id for chunk in added_chunks]

            # Add nodes to knowledge graph with assigned IDs
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
                selected_chunks = wave_reflection.select_top_chunks(
                    all_chunks=all_chunks,
                    knowledge_graph=knowledge_graph,
                    latest_chunk_ids=latest_chunk_ids,
                    capacity=working_memory.capacity,
                )
                # Update working memory with selected chunks
                working_memory._chunks = selected_chunks
                print(f"\nWave Reflection selected {len(selected_chunks)} chunks for working memory")
        else:
            print("\nNo chunks extracted from this segment")

    print(f"\n\n{'=' * 60}")
    print("=== Final Results ===")
    print(f"{'=' * 60}")
    print(f"Total text segments processed: {chunk_count}")
    print(f"Total chunks in knowledge graph: {knowledge_graph.number_of_nodes()}")
    print(f"Total connections: {knowledge_graph.number_of_edges()}")
    print(f"\nFinal working memory ({len(working_memory.get_chunks())} chunks):")
    for chunk in working_memory.get_chunks():
        print(f"  {chunk.id}. [{chunk.label}] {chunk.content}")

    # Save knowledge graph
    json_output_file = output_dir / "knowledge_graph.json"

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

    print(f"\nKnowledge graph saved to: {json_output_file}")
    print(f"  - {len(graph_data['nodes'])} nodes")
    print(f"  - {len(graph_data['edges'])} edges")

    # Detect snakes (thematic chains)
    print(f"\n{'=' * 60}")
    print("=== Snake Detection ===")
    print(f"{'=' * 60}")
    print("Splitting into connected components...")
    components = split_connected_components(knowledge_graph)
    print(f"Found {len(components)} connected component(s):")
    for i, comp in enumerate(components):
        print(f"  Component {i}: {len(comp.nodes())} nodes, {len(comp.edges())} edges")

    print("\nDetecting thematic chains (snakes)...")
    detector = SnakeDetector(
        min_cluster_size=2,  # Minimum snake length
        phase2_stop_ratio=0.15,  # Phase 2 stops at 15% of component nodes
    )

    all_snakes = []
    for i, component in enumerate(components):
        print(f"\nProcessing Component {i}:")
        component_snakes = detector.detect_snakes(component)
        all_snakes.extend(component_snakes)

    if all_snakes:
        print(f"\nFound {len(all_snakes)} snakes:")
        for i, snake in enumerate(all_snakes):
            first_node = knowledge_graph.nodes[snake[0]]
            last_node = knowledge_graph.nodes[snake[-1]]
            print(f"  Snake {i}: {len(snake)} nodes - {first_node['label']} → {last_node['label']}")

        # Save snakes
        snakes_output = output_dir / "snakes.json"
        snakes_data = []
        for i, snake in enumerate(all_snakes):
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
        print(f"\nSnakes saved to: {snakes_output}")

        # Summarize snakes with LLM
        print(f"\n{'=' * 60}")
        print("=== Snake Summarization ===")
        print(f"{'=' * 60}")
        prompt_file = data_dir / "snake_summary_prompt.jinja"
        summarizer = SnakeSummarizer(llm, prompt_file)

        print(f"Generating summaries for {len(all_snakes)} snakes...")
        snake_summaries = summarizer.summarize_all_snakes(all_snakes, knowledge_graph)

        # Save summaries
        summaries_output = output_dir / "snake_summaries.json"
        with open(summaries_output, "w", encoding="utf-8") as f:
            json.dump(snake_summaries, f, ensure_ascii=False, indent=2)

        print(f"\nSnake summaries saved to: {summaries_output}")
        print("\nSample summaries:")
        for summary in snake_summaries[:3]:  # Show first 3
            print(f"\n  Snake {summary['snake_id']}: {summary['first_label']} → {summary['last_label']}")
            print(f"  Summary: {summary['summary'][:100]}...")

        # Build snake-level graph with edges
        print(f"\n{'=' * 60}")
        print("=== Building Snake Graph ===")
        print(f"{'=' * 60}")

        # Calculate edge importance
        calculator = EdgeImportanceCalculator(knowledge_graph)
        edge_importance = calculator.compute_combined_importance()

        # Build snake graph
        builder = SnakeGraphBuilder()
        snake_graph = builder.build_snake_graph(all_snakes, knowledge_graph, edge_importance)

        print(f"Snake graph: {len(snake_graph.nodes())} snakes, {len(snake_graph.edges())} inter-snake edges")

        # Save snake edges
        edges_output = output_dir / "snake_edges.json"
        edges_data = []
        for edge in snake_graph.edges():
            snake_from, snake_to = edge
            edge_data = snake_graph.edges[edge]

            from_node = knowledge_graph.nodes[all_snakes[snake_from][0]]
            to_node = knowledge_graph.nodes[all_snakes[snake_to][0]]

            edges_data.append({
                "from_snake": snake_from,
                "to_snake": snake_to,
                "from_label": from_node["label"],
                "to_label": to_node["label"],
                "importance": edge_data["importance"],
                "internal_edge_count": edge_data["internal_edge_count"]
            })

        with open(edges_output, "w", encoding="utf-8") as f:
            json.dump(edges_data, f, ensure_ascii=False, indent=2)

        print(f"\nSnake edges saved to: {edges_output}")
        print(f"Sample edges:")
        for edge_info in edges_data[:5]:
            print(f"  Snake {edge_info['from_snake']} ({edge_info['from_label']}) -> Snake {edge_info['to_snake']} ({edge_info['to_label']})")
    else:
        print("\nNo snakes detected")


if __name__ == "__main__":
    main()
