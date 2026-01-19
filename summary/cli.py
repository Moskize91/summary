"""Command-line interface for topologization pipeline."""

import argparse
import sys
from pathlib import Path

from tiktoken import get_encoding

from .editor import compress_text
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
        "--phase2-ratio",
        type=float,
        default=0.15,
        help="Phase 2 stop ratio for snake detection (0-1)",
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
        phase2_stop_ratio=parsed_args.phase2_ratio,
    )

    try:
        intention = "压缩此书，重点关注朱元璋的心理变化。对于有关朱元璋心境改变和成长的关键点，必须整段保留，其他内容可以适当压缩。"

        # Run topologization
        topologization = topologize(
            intention=intention,
            input_file=parsed_args.input_file,
            workspace_path=parsed_args.workspace,
            config=config,
            llm=llm,
            encoding=get_encoding("o200k_base"),
        )

        # Run compression
        compressed_text = compress_text(
            topologization=topologization,
            intention=intention,
            llm=llm,
            compression_ratio=0.2,
            max_iterations=5,
            log_dir_path=log_dir / "compression" if log_dir is not None else None,
        )

        # Save compressed text
        output_path = parsed_args.workspace / "compressed.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(compressed_text)

        # Print summary
        print("\n" + "=" * 60)
        print("=== Pipeline Complete ===")
        print("=" * 60)
        print(f"Workspace: {parsed_args.workspace}")
        print(f"Compressed text saved to: {output_path}")
        print(f"Compressed text length: {len(compressed_text)} characters")

        return 0

    except Exception as e:
        print("\nError: Pipeline failed with exception:", file=sys.stderr)
        print(f"  {type(e).__name__}: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
