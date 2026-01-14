import json
from pathlib import Path
from typing import Any, cast

from pyvis.network import Network


def generate_html(json_path: Path, output_path: Path) -> None:
    """Generate interactive HTML visualization from knowledge graph JSON.

    Args:
        json_path: Path to the knowledge_graph.json file
        output_path: Path where the HTML file should be saved
    """
    # Load knowledge graph data
    with open(json_path, encoding="utf-8") as f:
        graph_data = json.load(f)

    # Create pyvis network
    net = Network(
        height="100vh",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
        font_color=cast(Any, "#000000"),
    )

    # Configure physics for better layout
    net.set_options(
        """
        {
            "physics": {
                "enabled": true,
                "stabilization": {
                    "enabled": true,
                    "iterations": 200
                },
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "springLength": 150,
                    "springConstant": 0.04
                }
            },
            "nodes": {
                "shape": "box",
                "margin": 10,
                "font": {"size": 14},
                "borderWidth": 2,
                "color": {
                    "border": "#2B7CE9",
                    "background": "#D2E5FF",
                    "highlight": {
                        "border": "#2B7CE9",
                        "background": "#FFF700"
                    }
                }
            },
            "edges": {
                "arrows": {
                    "to": {"enabled": true, "scaleFactor": 0.5}
                },
                "color": {"color": "#848484", "highlight": "#2B7CE9"},
                "smooth": {"enabled": true, "type": "dynamic"}
            }
        }
        """
    )

    # Add nodes
    for node in graph_data["nodes"]:
        node_id = node["id"]
        content = node["content"]
        # Truncate long content for display
        label = content if len(content) <= 50 else f"{content[:47]}..."
        net.add_node(
            node_id,
            label=f"{node_id}\n{label}",
            title=f"ID: {node_id}\n\n{content}",  # Full content in tooltip
        )

    # Add edges
    for edge in graph_data["edges"]:
        net.add_edge(edge["from"], edge["to"])

    # Generate HTML
    net.save_graph(str(output_path))

    print(f"Visualization saved to: {output_path}")
    print("Browse to the file to view the URL: file://" + str(output_path.resolve()))
    print("Open it in your browser to view the interactive graph.")
