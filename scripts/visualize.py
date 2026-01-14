"""Script to generate HTML visualization from knowledge graph JSON."""

from pathlib import Path

from dev.visualizer import generate_html


def main():
    """Generate visualization from the latest knowledge graph output."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    json_path = project_root / "output" / "knowledge_graph.json"
    html_path = project_root / "output" / "knowledge_graph.html"

    # Check if JSON exists
    if not json_path.exists():
        print(f"Error: Knowledge graph JSON not found at {json_path}")
        print("Please run the main extraction process first.")
        return

    # Generate HTML
    print(f"Reading knowledge graph from: {json_path}")
    generate_html(json_path, html_path)


if __name__ == "__main__":
    main()
