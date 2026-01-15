"""Main entry point for cognitive chunk extraction."""

import json
import shutil
from pathlib import Path

import networkx as nx

from .extractor import ChunkExtractor
from .llm import LLM
from .text_chunker import TextChunker
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

    # Initialize graph to store all chunks
    knowledge_graph = nx.DiGraph()

    print("=== Cognitive Chunk Extraction ===")
    print(f"Working memory capacity: {working_memory.capacity}")
    print(f"Input file: {input_file.name}")
    print(f"Output directory: {output_dir}")
    print(f"  - Logs: {log_dir}")
    print(f"Cache directory: {cache_dir} (persistent)\n")

    # Process text chunks
    chunk_count = 0
    max_chunks = 10  # Process first 10 chunks for testing

    for i, text_segment in enumerate(chunker.stream_chunks_from_file(input_file)):
        if i >= max_chunks:
            break

        chunk_count += 1
        print(f"\n{'=' * 60}")
        print(f"Processing segment {chunk_count}")
        print(f"{'=' * 60}")
        print(f"Text preview: {text_segment[:100]}...")
        print(f"\nCurrent working memory ({len(working_memory.get_chunks())} chunks):")
        print(working_memory.format_for_prompt())

        # Extract chunks
        print("\nExtracting cognitive chunks...")
        extraction_result = extractor.extract_chunks(text_segment, working_memory)

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

            # Add nodes to knowledge graph with assigned IDs
            for chunk in added_chunks:
                knowledge_graph.add_node(
                    chunk.id,
                    generation=chunk.generation,
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

    # Extract anchors using PageRank
    print(f"\n{'=' * 60}")
    print("=== Anchor Extraction ===")
    print(f"{'=' * 60}")

    if knowledge_graph.number_of_nodes() > 0:
        # Calculate PageRank scores
        pagerank_scores = nx.pagerank(knowledge_graph, alpha=0.85)

        # Sort nodes by PageRank score
        sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

        # Extract top N anchors (default: 10 or 20% of nodes, whichever is smaller)
        num_anchors = min(10, max(1, knowledge_graph.number_of_nodes() // 5))
        anchors = sorted_nodes[:num_anchors]

        print(f"Extracted {len(anchors)} anchors (top {num_anchors}):\n")
        for rank, (node_id, score) in enumerate(anchors, 1):
            node_data = knowledge_graph.nodes[node_id]
            in_degree = knowledge_graph.in_degree(node_id)
            out_degree = knowledge_graph.out_degree(node_id)
            print(
                f"{rank}. [{node_data['label']}] "
                f"(ID: {node_id}, Gen: {node_data['generation']}, "
                f"PageRank: {score:.4f}, In: {in_degree}, Out: {out_degree})"
            )
            print(f"   {node_data['content'][:100]}...")
    else:
        anchors = []
        print("No nodes in knowledge graph, cannot extract anchors.")

    # Save knowledge graph with anchors
    json_output_file = output_dir / "knowledge_graph.json"

    # Create anchor ID set for quick lookup
    anchor_ids = {node_id for node_id, _ in anchors} if anchors else set()

    graph_data = {
        "nodes": [
            {
                "id": n,
                "generation": knowledge_graph.nodes[n]["generation"],
                "label": knowledge_graph.nodes[n]["label"],
                "content": knowledge_graph.nodes[n]["content"],
                "is_anchor": n in anchor_ids,
            }
            for n in knowledge_graph.nodes()
        ],
        "edges": [{"from": u, "to": v} for u, v in knowledge_graph.edges()],
        "anchors": [
            {
                "rank": rank,
                "id": node_id,
                "pagerank_score": score,
                "generation": knowledge_graph.nodes[node_id]["generation"],
                "label": knowledge_graph.nodes[node_id]["label"],
                "content": knowledge_graph.nodes[node_id]["content"],
                "in_degree": knowledge_graph.in_degree(node_id),
                "out_degree": knowledge_graph.out_degree(node_id),
            }
            for rank, (node_id, score) in enumerate(anchors, 1)
        ],
    }

    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    print(f"\nKnowledge graph saved to: {json_output_file}")
    print(f"  - {len(graph_data['nodes'])} nodes")
    print(f"  - {len(graph_data['edges'])} edges")
    print(f"  - {len(graph_data['anchors'])} anchors")


if __name__ == "__main__":
    main()
