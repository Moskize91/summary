"""Command-line interface for topologization pipeline."""

import argparse
import sys
from pathlib import Path

from spacy.lang.xx import MultiLanguage
from tiktoken import Encoding, get_encoding

from .editor import compress_text
from .llm import LLM
from .topologization import TopologizationConfig, topologize


def _prepare_input(input_file: Path, encoding: Encoding, batch_size: int = 50000) -> list[list[tuple[int, str]]]:
    """Prepare input from file by splitting into sentences and counting tokens.

    Args:
        input_file: Path to input text file
        encoding: Tiktoken encoding for token counting
        batch_size: Maximum characters per batch for spaCy processing

    Returns:
        List with one chapter containing (token_count, sentence_text) tuples
    """
    # Load spaCy multilingual model with sentencizer (works for any language)
    nlp = MultiLanguage()
    nlp.add_pipe("sentencizer")

    # Read file and generate text batches
    text_buffer = ""
    sentences = []

    with open(input_file, encoding="utf-8") as f:
        for line in f:
            text_buffer += line

            # Process batch when buffer reaches batch_size
            if len(text_buffer) >= batch_size:
                doc = nlp(text_buffer)
                for sent in doc.sents:
                    sentence_text = sent.text.strip()
                    if sentence_text:
                        token_count = len(encoding.encode(sentence_text))
                        sentences.append((token_count, sentence_text))
                text_buffer = ""

    # Process remaining text
    if text_buffer:
        doc = nlp(text_buffer)
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if sentence_text:
                token_count = len(encoding.encode(sentence_text))
                sentences.append((token_count, sentence_text))

    # Return as single chapter
    return [sentences]


def main(args: list[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        args: Command-line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Cognitive chunk extraction and thematic chain detection pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input text file to process",
    )

    # Optional arguments
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("format.json"),
        help="Path to LLM config file (format.json)",
    )

    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace"),
        help="Workspace directory for storing fragments and database",
    )

    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Log directory for LLM logs (default: logs/)",
    )

    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("cache"),
        help="Cache directory for LLM cache",
    )

    # Processing parameters
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=40,
        help="Maximum number of text chunks to process (0 for unlimited)",
    )

    parser.add_argument(
        "--fragment-tokens",
        type=int,
        default=800,
        help="Maximum token count per text fragment",
    )

    parser.add_argument(
        "--memory-capacity",
        type=int,
        default=7,
        help="Working memory capacity (number of chunks)",
    )

    parser.add_argument(
        "--decay-factor",
        type=float,
        default=0.5,
        help="Generation decay factor for Wave Reflection (0-1)",
    )

    # Snake detection parameters
    parser.add_argument(
        "--min-snake-size",
        type=int,
        default=2,
        help="Minimum number of nodes in a snake",
    )

    parser.add_argument(
        "--snake-tokens",
        type=int,
        default=700,
        help="Maximum tokens allowed in a snake",
    )

    # Parse arguments
    if args is None:
        args = sys.argv[1:]

    parsed_args = parser.parse_args(args)

    # Validate input file
    if not parsed_args.input_file.exists():
        print(f"Error: Input file not found: {parsed_args.input_file}", file=sys.stderr)
        return 1

    # Validate config file
    if not parsed_args.config.exists():
        print(f"Error: Config file not found: {parsed_args.config}", file=sys.stderr)
        return 1

    # Setup log directory
    log_dir = parsed_args.log if parsed_args.log is not None else Path("logs")

    # Setup cache directory
    cache_dir = parsed_args.cache

    # Handle max_chunks = 0 (unlimited)
    max_chunks = parsed_args.max_chunks if parsed_args.max_chunks > 0 else None

    # Setup data directory for templates
    data_dir = Path(__file__).parent / "data"

    # Create LLM instance
    llm = LLM(
        config_path=parsed_args.config,
        data_dir_path=data_dir,
        log_dir_path=log_dir / "llm" if log_dir is not None else None,
        cache_dir_path=cache_dir,
    )

    # Create pipeline configuration
    config = TopologizationConfig(
        max_fragment_tokens=parsed_args.fragment_tokens,
        working_memory_capacity=parsed_args.memory_capacity,
        generation_decay_factor=parsed_args.decay_factor,
        max_chunks=max_chunks,
        min_cluster_size=parsed_args.min_snake_size,
        snake_tokens=parsed_args.snake_tokens,
    )

    try:
        intention: str = (
            "压缩此书，重点关注别人对朱元璋的看法，别人怎么对待朱元璋，以及朱元璋被他人以不同态度对待的反应。"
            "对于他人第一次见朱元璋的评价、反应、表情，必须一字不漏地原文保留。社会背景方面可以压缩和删节。"
        )

        # Prepare input: read file, split sentences, count tokens
        encoding = get_encoding("o200k_base")
        print(f"Processing input file: {parsed_args.input_file}")
        input_data = _prepare_input(parsed_args.input_file, encoding)
        print(f"Prepared {len(input_data[0])} sentences")

        # Run topologization
        topologization = topologize(
            intention=intention,
            input=input_data,
            workspace_path=parsed_args.workspace,
            config=config,
            llm=llm,
            encoding=encoding,
        )

        # Run compression for each chapter and group
        print("\n" + "=" * 60)
        print("=== Compression Stage ===")
        print("=" * 60)

        compressed_dir = parsed_args.workspace / "compressed"
        compressed_dir.mkdir(parents=True, exist_ok=True)

        all_chapter_ids = topologization.get_all_chapter_ids()
        print(f"\nFound {len(all_chapter_ids)} chapters to compress")

        for chapter_id in all_chapter_ids:
            group_ids = topologization.get_group_ids_for_chapter(chapter_id)
            print(f"\nChapter {chapter_id}: {len(group_ids)} groups")

            # Create chapter directory
            chapter_dir = compressed_dir / f"chapter-{chapter_id}"
            chapter_dir.mkdir(parents=True, exist_ok=True)

            for group_id in group_ids:
                print(f"\nCompressing Chapter {chapter_id}, Group {group_id}...")

                # Compress this group
                compressed_text = compress_text(
                    topologization=topologization,
                    chapter_id=chapter_id,
                    group_id=group_id,
                    intention=intention,
                    llm=llm,
                    compression_ratio=0.2,
                    max_iterations=10,
                    max_clues=6,
                    log_dir_path=log_dir / "compression" if log_dir is not None else None,
                )

                # Save compressed text
                output_path = chapter_dir / f"group-{group_id}.txt"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(compressed_text)

                print(f"✓ Saved to: {output_path}")
                print(f"  Length: {len(compressed_text)} characters")

        # Print summary
        print("\n" + "=" * 60)
        print("=== Pipeline Complete ===")
        print("=" * 60)
        print(f"Workspace: {parsed_args.workspace}")
        print(f"Compressed texts saved to: {compressed_dir}")

        # Count total compressed files
        total_groups = sum(len(topologization.get_group_ids_for_chapter(cid)) for cid in all_chapter_ids)
        print(f"Total compressed groups: {total_groups}")

        return 0

    except Exception as e:
        print("\nError: Pipeline failed with exception:", file=sys.stderr)
        print(f"  {type(e).__name__}: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
