"""Command-line interface for topologization pipeline."""

import argparse
import sys
from pathlib import Path

from .llm import LLM
from .template import create_env
from .topologization.core import PipelineConfig, TopologizationPipeline


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
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory for results",
    )

    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Log directory for LLM logs (default: output/logs)",
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
        "--chunk-length",
        type=int,
        default=800,
        help="Maximum character length per text chunk",
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
        default=0.68,
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

    # Setup log directory (default to output/logs if not specified)
    log_dir = parsed_args.log
    if log_dir is None:
        log_dir = parsed_args.output / "logs"

    # Setup cache directory
    cache_dir = parsed_args.cache

    # Handle max_chunks = 0 (unlimited)
    max_chunks = parsed_args.max_chunks if parsed_args.max_chunks > 0 else None

    # Setup prompt file paths
    data_dir = Path(__file__).parent / "data"
    extraction_prompt_file = data_dir / "extraction_prompt.jinja"
    snake_summary_prompt_file = data_dir / "snake_summary_prompt.jinja"

    # Validate prompt files
    if not extraction_prompt_file.exists():
        print(f"Error: Extraction prompt file not found: {extraction_prompt_file}", file=sys.stderr)
        return 1

    if not snake_summary_prompt_file.exists():
        print(f"Error: Snake summary prompt file not found: {snake_summary_prompt_file}", file=sys.stderr)
        return 1

    # Create LLM instance
    llm = LLM(
        config_path=parsed_args.config,
        log_dir_path=log_dir,
        cache_dir_path=cache_dir,
    )

    # Create Jinja2 environment
    jinja_env = create_env(data_dir)

    # Create pipeline configuration
    config = PipelineConfig(
        input_file=parsed_args.input_file,
        output_dir=parsed_args.output,
        extraction_prompt_file=extraction_prompt_file,
        snake_summary_prompt_file=snake_summary_prompt_file,
        max_chunk_length=parsed_args.chunk_length,
        working_memory_capacity=parsed_args.memory_capacity,
        generation_decay_factor=parsed_args.decay_factor,
        max_chunks=max_chunks,
        min_cluster_size=parsed_args.min_snake_size,
        phase2_stop_ratio=parsed_args.phase2_ratio,
        clear_output_on_start=True,
        save_intermediate_results=True,
    )

    try:
        # Run pipeline
        pipeline = TopologizationPipeline(config, llm, jinja_env)
        result = pipeline.run()

        # Print summary
        print("\n" + "=" * 60)
        print("=== Pipeline Complete ===")
        print("=" * 60)
        print(f"Total chunks: {len(result.all_chunks)}")
        print(f"Total snakes: {len(result.snakes)}")
        print("\nOutput files:")
        for name, path in result.output_files.items():
            print(f"  - {name}: {path}")

        return 0

    except Exception as e:
        print("\nError: Pipeline failed with exception:", file=sys.stderr)
        print(f"  {type(e).__name__}: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
