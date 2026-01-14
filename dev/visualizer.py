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

    # Configure hierarchical layout (top to bottom based on generation)
    net.set_options(
        """
        {
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "UD",
                    "sortMethod": "directed",
                    "levelSeparation": 150,
                    "nodeSpacing": 200
                }
            },
            "physics": {
                "enabled": false
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
                "smooth": {"enabled": true, "type": "cubicBezier"}
            }
        }
        """
    )

    # Add nodes with generation-based levels
    for node in graph_data["nodes"]:
        node_id = node["id"]
        generation = node["generation"]
        label = node["label"]
        content = node["content"]
        is_anchor = node.get("is_anchor", False)

        # Different colors for anchor vs regular nodes
        if is_anchor:
            bg_color = "#FFEBEE"  # Light red background for anchors
            border_width = 3
            anchor_marker = "⚓ "
        else:
            bg_color = "#D2E5FF"  # Light blue background for regular nodes
            border_width = 2
            anchor_marker = ""

        # Display ID and label on node, full content in tooltip
        anchor_prefix = "[ANCHOR] " if is_anchor else ""
        tooltip = f"{anchor_prefix}ID: {node_id}\nGeneration: {generation}\nLabel: {label}\n\nContent:\n{content}"

        net.add_node(
            node_id,
            label=f"{anchor_marker}{node_id}\n{label}",
            title=tooltip,
            level=generation,  # Use generation as hierarchical level
            color=bg_color,
            borderWidth=border_width,
        )

    # Add edges
    for edge in graph_data["edges"]:
        net.add_edge(edge["from"], edge["to"])

    # Generate HTML
    net.save_graph(str(output_path))

    print(f"Visualization saved to: {output_path}")
    print("Browse to the file to view the URL: file://" + str(output_path.resolve()))
    print("Open it in your browser to view the interactive graph.")
