import shutil
import sys
from pathlib import Path

from summary.cli import main as cli_main

if __name__ == "__main__":
    # Setup paths based on original configuration
    project_root = Path(__file__).parent.resolve()
    input_file = project_root / "summary" / "data" / "明朝那些事儿.txt"
    config_file = project_root / "format.json"
    workspace_dir = project_root / "workspace"
    output_path = project_root / "output"
    log_dir = output_path / "logs"
    cache_dir = project_root / "cache"

    # Ensure workspace_dir is an empty folder on each startup
    for path in (workspace_dir, output_path):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    # Build command-line arguments
    args = [
        str(input_file),
        "--config",
        str(config_file),
        "--workspace",
        str(workspace_dir),
        "--log",
        str(log_dir),
        "--cache",
        str(cache_dir),
        "--max-chunks",
        "10",
    ]

    # Run CLI
    sys.exit(cli_main(args))
