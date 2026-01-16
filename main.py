import sys
from pathlib import Path

from summary.cli import main as cli_main

if __name__ == "__main__":
    # Setup paths based on original configuration
    project_root = Path(__file__).parent
    data_dir = project_root / "summary" / "data"
    input_file = data_dir / "明朝那些事儿.txt"
    config_file = project_root / "format.json"
    output_dir = project_root / "output"
    cache_dir = project_root / "cache"

    # Build command-line arguments
    args = [
        str(input_file),
        "--config",
        str(config_file),
        "--output",
        str(output_dir),
        "--cache",
        str(cache_dir),
        "--max-chunks",
        "40",  # Process first 40 chunks for testing
    ]

    # Run CLI
    sys.exit(cli_main(args))
