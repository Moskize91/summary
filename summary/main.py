"""Main entry point for cognitive chunk extraction."""

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
    log_dir = project_root / "logs"

    # Initialize components
    chunker = TextChunker(max_chunk_length=800, batch_size=50000)
    llm = LLM(config_path=config_file, log_dir_path=log_dir)
    extractor = ChunkExtractor(llm, prompt_file)
    working_memory = WorkingMemory(capacity=7)

    # Initialize graph to store all chunks
    knowledge_graph = nx.DiGraph()

    print("=== Cognitive Chunk Extraction ===")
    print(f"Working memory capacity: {working_memory.capacity}")
    print(f"Input file: {input_file.name}\n")

    # Process text chunks
    chunk_count = 0
    max_chunks = 10  # Process first 10 chunks for testing

    for i, text_segment in enumerate(chunker.stream_chunks_from_file(input_file)):
        if i >= max_chunks:
            break

        chunk_count += 1
        print(f"\n{'='*60}")
        print(f"Processing segment {chunk_count}")
        print(f"{'='*60}")
        print(f"Text preview: {text_segment[:100]}...")
        print(f"\nCurrent working memory ({len(working_memory.get_chunks())} chunks):")
        print(working_memory.format_for_prompt())

        # Extract chunks
        print("\nExtracting cognitive chunks...")
        new_chunks = extractor.extract_chunks(text_segment, working_memory)

        if new_chunks:
            print(f"\nExtracted {len(new_chunks)} new chunks:")
            for chunk in new_chunks:
                print(f"  - {chunk.content[:80]}... (links: {chunk.links})")

            # Add to working memory
            working_memory.add_chunks(new_chunks)

            # Add to knowledge graph
            for chunk in new_chunks:
                knowledge_graph.add_node(chunk.id, content=chunk.content)
                for link_id in chunk.links:
                    if knowledge_graph.has_node(link_id):
                        knowledge_graph.add_edge(link_id, chunk.id)
        else:
            print("\nNo chunks extracted (LLM failed or returned empty)")

    print(f"\n\n{'='*60}")
    print("=== Final Results ===")
    print(f"{'='*60}")
    print(f"Total text segments processed: {chunk_count}")
    print(f"Total chunks in knowledge graph: {knowledge_graph.number_of_nodes()}")
    print(f"Total connections: {knowledge_graph.number_of_edges()}")
    print(f"\nFinal working memory ({len(working_memory.get_chunks())} chunks):")
    for chunk in working_memory.get_chunks():
        print(f"  {chunk.id}. {chunk.content}")

    # Save knowledge graph
    output_file = project_root / "knowledge_graph.json"
    graph_data = {
        "nodes": [{"id": n, "content": knowledge_graph.nodes[n]["content"]} for n in knowledge_graph.nodes()],
        "edges": [{"from": u, "to": v} for u, v in knowledge_graph.edges()],
    }

    import json

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    print(f"\nKnowledge graph saved to: {output_file}")


if __name__ == "__main__":
    main()

