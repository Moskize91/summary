import json
import shutil
from pathlib import Path

from summary import summary

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

    # Validate input file
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        exit(1)

    # Validate and load config file
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        exit(1)

    with open(config_file, encoding="utf-8") as f:
        llm_config = json.load(f)

    # Ensure workspace_dir is an empty folder on each startup
    for path in (workspace_dir, output_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    # Define intention (hardcoded for now, could be made configurable)
    intention = (
        "压缩此书，重点关注别人对朱元璋的看法，别人怎么对待朱元璋，以及朱元璋被他人以不同态度对待的反应。"
        "对于他人第一次见朱元璋的评价、反应、表情，必须一字不漏地原文保留。社会背景方面可以压缩和删节。"
    )

    # Call summary function
    summary(
        intention=intention,
        input_epub_file=input_file,
        output_epub_file=output_file,
        llm_api_key=llm_config["key"],
        llm_base_url=llm_config["url"],
        llm_model=llm_config["model"],
        workspace_path=workspace_dir,
        log_dir=log_dir,
        cache_dir=cache_dir,
        llm_timeout=llm_config.get("timeout", 360.0),
        llm_temperature=llm_config.get("temperature", 0.6),
        llm_top_p=llm_config.get("top_p", 0.6),
    )
