"""Script to generate SVG visualization from knowledge graph JSON."""

from pathlib import Path

from dev.visualizer import generate_svg


def main():
    """Generate visualization from the latest knowledge graph output."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    json_path = project_root / "output" / "knowledge_graph.json"
    svg_path = project_root / "output" / "knowledge_graph"

    # Check if JSON exists
    if not json_path.exists():
        print(f"Error: Knowledge graph JSON not found at {json_path}")
        print("Please run the main extraction process first.")
        return

    # Generate SVG
    print(f"Reading knowledge graph from: {json_path}")
    generate_svg(json_path, svg_path)


if __name__ == "__main__":
    main()
