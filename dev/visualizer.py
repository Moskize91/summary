import json
import re
from pathlib import Path

from graphviz import Digraph


def generate_svg(json_path: Path, output_path: Path) -> None:
    """Generate SVG visualization with HTML wrapper from knowledge graph JSON.

    Args:
        json_path: Path to the knowledge_graph.json file
        output_path: Path where the files should be saved (without extension)
    """
    # Load knowledge graph data
    with open(json_path, encoding="utf-8") as f:
        graph_data = json.load(f)

    # Create graphviz digraph
    dot = Digraph(comment="Knowledge Graph", format="svg")

    # Configure graph attributes for top-to-bottom hierarchical layout
    dot.attr(rankdir="TB")  # Top to Bottom
    dot.attr(splines="ortho")  # Orthogonal edges for cleaner look
    dot.attr(nodesep="1.4")  # Horizontal spacing between nodes (increased for clarity)
    dot.attr(ranksep="1.2")  # Vertical spacing between ranks
    dot.attr(center="true")  # Center the graph
    dot.attr(margin="0.5")  # Add margin around the graph

    # Configure default node attributes
    dot.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fillcolor="#D2E5FF",
        color="#2B7CE9",
        fontname="Arial",
        fontsize="12",
        margin="0.3,0.2",
    )

    # Configure default edge attributes
    dot.attr("edge", color="#848484", arrowsize="0.8")

    # Add nodes
    for node in graph_data["nodes"]:
        node_id = str(node["id"])
        sentence_id = node["sentence_id"]
        label = node["label"]
        content = node["content"]

        # Node label: ID + label
        node_label = f"{node_id}\\n{label}"

        # Tooltip: full information
        tooltip = f"ID: {node_id}\\nSentence ID: {sentence_id}\\nLabel: {label}\\n\\nContent:\\n{content}"

        dot.node(node_id, label=node_label, tooltip=tooltip)

    # Sort nodes by sentence_id to maintain temporal order
    sorted_nodes = sorted(graph_data["nodes"], key=lambda n: n["sentence_id"])

    # Add invisible edges between consecutive chunks to maintain ordering
    for i in range(len(sorted_nodes) - 1):
        current_id = str(sorted_nodes[i]["id"])
        next_id = str(sorted_nodes[i + 1]["id"])
        # Add invisible constraint edge
        dot.edge(current_id, next_id, style="invis", constraint="true")

    # Add edges (reverse direction so earlier chunks appear at top)
    for edge in graph_data["edges"]:
        from_id = str(edge["from"])
        to_id = str(edge["to"])
        # Reverse edge direction: earlier -> later (so earlier appears at top)
        dot.edge(to_id, from_id)

    # Render to SVG
    output_path_str = str(output_path.with_suffix(""))  # Remove extension if present
    dot.render(output_path_str, cleanup=True)

    svg_path = Path(f"{output_path_str}.svg")

    # Generate HTML wrapper with interactive tooltip
    html_path = Path(f"{output_path_str}.html")
    _generate_html_wrapper(svg_path, html_path, graph_data)

    print(f"Visualization saved to: {html_path}")
    print(f"Open it in your browser to view: file://{html_path.resolve()}")


def _generate_html_wrapper(svg_path: Path, html_path: Path, graph_data: dict) -> None:
    """Generate HTML file that embeds SVG with interactive tooltips.

    Args:
        svg_path: Path to the SVG file
        html_path: Path where HTML file should be saved
        graph_data: Knowledge graph data for tooltips
    """
    # Read SVG content
    with open(svg_path, encoding="utf-8") as f:
        svg_content = f.read()

    # Remove xlink:title attributes to prevent native browser tooltips
    svg_content = re.sub(r'\s*xlink:title="[^"]*"', "", svg_content)

    # Build node data mapping for JavaScript
    node_data = {}
    for node in graph_data["nodes"]:
        node_id = str(node["id"])
        node_data[node_id] = {
            "id": node_id,
            "sentence_id": node["sentence_id"],
            "label": node["label"],
            "content": node["content"],
        }

    # Create HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            min-height: 100vh;
        }}

        #container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 20px;
            max-width: 95%;
        }}

        #svg-container {{
            display: flex;
            justify-content: center;
            overflow: auto;
        }}

        /* Hide SVG title elements to prevent native tooltips */
        svg title {{
            display: none;
        }}

        #tooltip {{
            position: fixed;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 14px;
            line-height: 1.5;
            max-width: 400px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 1000;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}

        #tooltip.show {{
            opacity: 1;
        }}

        .tooltip-label {{
            font-weight: bold;
            color: #FFF700;
            font-size: 15px;
            margin-bottom: 2px;
            line-height: 1.3;
        }}

        .tooltip-meta {{
            color: #aaa;
            font-size: 11px;
            margin-bottom: 6px;
            line-height: 1.2;
        }}

        .tooltip-content {{
            border-top: 1px solid #555;
            padding-top: 6px;
            margin-top: 4px;
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="svg-container">
            {svg_content}
        </div>
    </div>
    <div id="tooltip"></div>

    <script>
        // Node data from Python
        const nodeData = {json.dumps(node_data, ensure_ascii=False)};

        // Get tooltip element
        const tooltip = document.getElementById('tooltip');

        // Find all node elements in SVG
        const svg = document.querySelector('svg');
        const nodes = svg.querySelectorAll('g.node');

        nodes.forEach(node => {{
            const title = node.querySelector('title');
            if (!title) return;

            const nodeId = title.textContent.trim();
            const data = nodeData[nodeId];
            if (!data) return;

            // Add hover listeners
            node.addEventListener('mouseenter', (e) => {{
                const tooltipHTML = `
                    <div class="tooltip-label">${{data.label}}</div>
                    <div class="tooltip-meta">ID: ${{data.id}} | Sentence ID: ${{data.sentence_id}}</div>
                    <div class="tooltip-content">${{data.content}}</div>
                `;
                tooltip.innerHTML = tooltipHTML;
                tooltip.classList.add('show');
            }});

            node.addEventListener('mousemove', (e) => {{
                tooltip.style.left = (e.clientX + 15) + 'px';
                tooltip.style.top = (e.clientY + 15) + 'px';
            }});

            node.addEventListener('mouseleave', () => {{
                tooltip.classList.remove('show');
            }});
        }});
    </script>
</body>
</html>
"""

    # Write HTML file
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
