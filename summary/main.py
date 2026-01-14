"""Main entry point for summarizing text using LLM."""

import json
from pathlib import Path

from openai import OpenAI

from .template import create_env
from .text_chunker import TextChunker


class LLMSummarizer:
    """Summarizes text chunks using an LLM."""

    def __init__(self, config_path: Path):
        """Initialize the LLM summarizer.

        Args:
            config_path: Path to the configuration JSON file
        """
        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        self.client = OpenAI(
            api_key=self.config["key"],
            base_url=self.config["url"],
            timeout=self.config.get("timeout", 360.0),
        )
        self.model = self.config["model"]
        self.temperature = self.config.get("temperature", 0.6)
        self.top_p = self.config.get("top_p", 0.6)

    def load_system_prompt(self, prompt_template_path: Path, **kwargs) -> str:
        """Load and render the system prompt from a Jinja template.

        Args:
            prompt_template_path: Path to the prompt.jinja file
            **kwargs: Variables to pass to the template

        Returns:
            Rendered system prompt
        """
        env = create_env(prompt_template_path.parent)
        template = env.get_template(prompt_template_path.name)
        return template.render(**kwargs)

    def summarize_chunk(self, chunk: str, system_prompt: str) -> str | None:
        """Summarize a single chunk of text.

        Args:
            chunk: Text chunk to summarize
            system_prompt: System prompt for the LLM

        Returns:
            Summary of the chunk, or None if the request failed
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            return None


def main():
    """Main function to run the summarization process."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = Path(__file__).parent / "data"
    input_file = data_dir / "明朝那些事儿.txt"
    config_file = project_root / "format.json"
    prompt_file = data_dir / "prompt.jinja"

    # Initialize components
    chunker = TextChunker(max_chunk_length=2000, batch_size=50000)
    summarizer = LLMSummarizer(config_file)

    # Load system prompt (you can pass variables to the template here)
    system_prompt = summarizer.load_system_prompt(prompt_file)
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
        summary = summarizer.summarize_chunk(chunk, system_prompt)

        if summary:
            print(f"Summary: {summary}\n")
        else:
            print("Failed to get summary\n")

    print("=== Processing complete ===")
