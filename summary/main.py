"""Main entry point for summarizing text using LLM."""

from pathlib import Path

from .llm import LLM
from .text_chunker import TextChunker


def main():
    """Main function to run the summarization process."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = Path(__file__).parent / "data"
    input_file = data_dir / "明朝那些事儿.txt"
    config_file = project_root / "format.json"
    prompt_file = data_dir / "prompt.jinja"
    log_dir = project_root / "logs"

    # Initialize components
    chunker = TextChunker(max_chunk_length=1500, batch_size=50000)
    llm = LLM(config_path=config_file, log_dir_path=log_dir)

    # Load system prompt (you can pass variables to the template here)
    system_prompt = llm.load_system_prompt(prompt_file)
    print(f"System prompt loaded: {system_prompt[:100]}...")

    # Process only the first 3 chunks for testing (streaming from file)
    print("\n=== Processing text chunks (streaming) ===\n")
    for i, chunk in enumerate(chunker.stream_chunks_from_file(input_file)):
        if i >= 3:  # Only process first 3 chunks
            break

        print(f"--- Chunk {i + 1} ---")
        print(f"Length: {len(chunk)} characters")
        print(f"Preview: {chunk[:100]}...")

        # Send to LLM
        print("Sending to LLM...")
        summary = llm.request(system_prompt=system_prompt, user_message=chunk)

        if summary:
            print(f"Summary: {summary}\n")
        else:
            print("Failed to get summary\n")

    print("=== Processing complete ===")


if __name__ == "__main__":
    main()

