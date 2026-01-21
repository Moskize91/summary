import shutil
import sys
from pathlib import Path

from summary.cli import main as cli_main

if __name__ == "__main__":
    # Setup paths based on original configuration
    project_root = Path(__file__).parent.resolve()
    input_file = project_root / "tests" / "assets" / "明朝那些事儿.epub"
    output_file = project_root / "output" / "明朝那些事儿_compressed.epub"
    config_file = project_root / "format.json"
    workspace_dir = project_root / "workspace"
    output_dir = project_root / "output"
    log_dir = output_dir / "logs"
    cache_dir = project_root / "cache"

    # Ensure workspace_dir is an empty folder on each startup
    for path in (workspace_dir, output_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    # Build command-line arguments
    args = [
        str(input_file),
        str(output_file),
        "--config",
        str(config_file),
        "--workspace",
        str(workspace_dir),
        "--log",
        str(log_dir),
        "--cache",
        str(cache_dir),
    ]

    # Run CLI
    sys.exit(cli_main(args))
