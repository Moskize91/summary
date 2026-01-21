"""Command-line interface for topologization pipeline."""

import argparse
import sys
from pathlib import Path

from tiktoken import get_encoding

from .editor import compress_text
from .epub import read_epub_sentences, write_epub
from .llm import LLM
from .topologization import TopologizationConfig, topologize


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
        help="Input EPUB file to process",
    )

    parser.add_argument(
        "output_file",
        type=Path,
        help="Output EPUB file path",
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
        min_cluster_size=parsed_args.min_snake_size,
        snake_tokens=parsed_args.snake_tokens,
    )

    try:
        intention: str = (
            "压缩此书，重点关注别人对朱元璋的看法，别人怎么对待朱元璋，以及朱元璋被他人以不同态度对待的反应。"
            "对于他人第一次见朱元璋的评价、反应、表情，必须一字不漏地原文保留。社会背景方面可以压缩和删节。"
        )

        # Prepare input: read EPUB and extract sentences
        encoding = get_encoding("o200k_base")
        print(f"Processing input EPUB: {parsed_args.input_file}")

        # Read EPUB and collect chapter data
        chapter_titles = []
        input_data = []

        for chapter_title, chapter_sentences in read_epub_sentences(parsed_args.input_file, encoding):
            chapter_titles.append(chapter_title)
            sentences_list = list(chapter_sentences)
            input_data.append(sentences_list)
            print(f"  Chapter: {chapter_title} ({len(sentences_list)} sentences)")

        print(f"Total chapters: {len(input_data)}")

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

        all_chapter_ids = topologization.get_all_chapter_ids()
        print(f"\nFound {len(all_chapter_ids)} chapters to compress")

        # Collect compressed chapters for EPUB output
        compressed_chapters = []

        for chapter_id in all_chapter_ids:
            group_ids = topologization.get_group_ids_for_chapter(chapter_id)
            print(f"\nChapter {chapter_id}: {chapter_titles[chapter_id]}")
            print(f"  Groups: {len(group_ids)}")

            # Compress all groups in this chapter
            chapter_compressed_texts = []

            for group_id in group_ids:
                print(f"  Compressing Group {group_id}...")

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

                chapter_compressed_texts.append(compressed_text)
                print(f"    ✓ Length: {len(compressed_text)} characters")

            # Merge all groups in this chapter with double newlines
            full_chapter_text = "\n\n".join(chapter_compressed_texts)
            compressed_chapters.append((chapter_titles[chapter_id], [full_chapter_text]))

        # Write output EPUB
        print("\n" + "=" * 60)
        print("=== Writing Output EPUB ===")
        print("=" * 60)

        # Extract book title from input filename (remove .epub extension)
        book_title = parsed_args.input_file.stem + " (压缩版)"

        write_epub(
            chapters=compressed_chapters,
            output_path=parsed_args.output_file,
            book_title=book_title,
            author="AI压缩",
            language="zh",
        )

        # Print summary
        print("\n" + "=" * 60)
        print("=== Pipeline Complete ===")
        print("=" * 60)
        print(f"Workspace: {parsed_args.workspace}")
        print(f"Output EPUB: {parsed_args.output_file}")
        print(f"Total chapters: {len(compressed_chapters)}")

        return 0

    except Exception as e:
        print("\nError: Pipeline failed with exception:", file=sys.stderr)
        print(f"  {type(e).__name__}: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
