"""Visualize detected snakes with color-coded clusters."""

import json
import re
from pathlib import Path

import networkx as nx
from graphviz import Digraph


def visualize_snakes(
    graph: nx.DiGraph,
    snakes: list[list[int]],
    output_path: Path,
    graph_data: dict,
    edge_importance: dict[frozenset, float] | None = None,
) -> None:
    """Generate visualization with color-coded snakes and edge importance.

    Args:
        graph: NetworkX graph
        snakes: List of detected snakes
        output_path: Output file path (without extension)
        graph_data: Original graph data dict for tooltips
        edge_importance: Optional dict mapping edge (frozenset) to importance score [0.0, 1.0]
    """
    # Create graphviz digraph
    dot = Digraph(comment="Knowledge Graph with Snakes", format="svg")

    # Configure graph attributes
    dot.attr(rankdir="TB")
    dot.attr(splines="ortho")
    dot.attr(nodesep="2.0")
    dot.attr(ranksep="1.2")
    dot.attr(center="true")
    dot.attr(margin="0.5")

    # Configure default node attributes
    dot.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fontname="Arial",
        fontsize="12",
        margin="0.3,0.2",
    )

    # Configure default edge attributes
    dot.attr("edge", color="#848484", arrowsize="0.8")

    # Define color palette for snakes
    colors = [
        "#FF6B6B",  # Red
        "#4ECDC4",  # Teal
        "#45B7D1",  # Blue
        "#FFA07A",  # Light Salmon
        "#98D8C8",  # Mint
        "#F7DC6F",  # Yellow
        "#BB8FCE",  # Purple
        "#85C1E2",  # Sky Blue
        "#F8B88B",  # Peach
        "#A3E4D7",  # Aqua
    ]

    # Build node-to-snake mapping
    node_to_snake = {}
    for snake_id, snake in enumerate(snakes):
        for node_id in snake:
            node_to_snake[node_id] = snake_id

    # Add nodes with snake colors
    for node in graph_data["nodes"]:
        node_id = node["id"]
        sentence_id = node["sentence_id"]
        label = node["label"]
        content = node["content"]

        # Node label: ID + label
        node_label = f"{node_id}\\n{label}"

        # Tooltip
        tooltip = f"ID: {node_id}\\nSentence ID: {sentence_id}\\nLabel: {label}\\n\\nContent:\\n{content}"

        # Color based on snake membership
        if node_id in node_to_snake:
            snake_id = node_to_snake[node_id]
            color = colors[snake_id % len(colors)]
            # Add snake info to tooltip
            tooltip = f"[Snake {snake_id}] " + tooltip
            # Use bright fill color with darker border
            dot.node(
                str(node_id),
                label=node_label,
                tooltip=tooltip,
                fillcolor=color + "80",  # Semi-transparent
                color=color,
                penwidth="3",
            )
        else:
            # Non-snake nodes: default gray
            dot.node(
                str(node_id),
                label=node_label,
                tooltip=tooltip,
                fillcolor="#E0E0E0",
                color="#999999",
            )

    # Infer positions for nodes with sentence_id=0
    def infer_sentence_id(node_id: int, graph_data: dict) -> int:
        """Infer sentence_id for nodes with sentence_id=0 based on graph topology."""
        node = next(n for n in graph_data["nodes"] if n["id"] == node_id)
        if node["sentence_id"] != 0:
            return node["sentence_id"]

        # Find predecessors and successors
        preds = [e["from"] for e in graph_data["edges"] if e["to"] == node_id]
        succs = [e["to"] for e in graph_data["edges"] if e["from"] == node_id]

        # Get sentence_ids of neighbors
        neighbor_ids = []
        for pred_id in preds:
            pred_node = next((n for n in graph_data["nodes"] if n["id"] == pred_id), None)
            if pred_node and pred_node["sentence_id"] != 0:
                neighbor_ids.append(pred_node["sentence_id"])

        for succ_id in succs:
            succ_node = next((n for n in graph_data["nodes"] if n["id"] == succ_id), None)
            if succ_node and succ_node["sentence_id"] != 0:
                neighbor_ids.append(succ_node["sentence_id"])

        if neighbor_ids:
            # Use average of neighbors' sentence_ids
            return sum(neighbor_ids) / len(neighbor_ids)
        else:
            # Fallback: use node_id as proxy
            return node_id * 1000  # Large number to push to end

    # Sort nodes by inferred sentence_id for temporal ordering
    sorted_nodes = sorted(graph_data["nodes"], key=lambda n: infer_sentence_id(n["id"], graph_data))

    # Add invisible edges for temporal ordering
    for i in range(len(sorted_nodes) - 1):
        current_id = str(sorted_nodes[i]["id"])
        next_id = str(sorted_nodes[i + 1]["id"])
        dot.edge(current_id, next_id, style="invis", constraint="true")

    # Add visible edges (reversed direction for top-to-bottom layout)
    for edge in graph_data["edges"]:
        from_id = str(edge["from"])
        to_id = str(edge["to"])

        # Check if both nodes are in the same snake
        from_node_id = edge["from"]
        to_node_id = edge["to"]

        # Determine edge color and width based on snake membership and importance
        edge_key = frozenset([from_node_id, to_node_id])
        importance = edge_importance.get(edge_key, 0.5) if edge_importance else 0.5

        # Map importance to color (0.0=light gray, 1.0=black)
        # Use a color gradient from #CCCCCC to #000000
        gray_value = int(204 * (1.0 - importance))  # 204 = 0xCC
        edge_color = f"#{gray_value:02x}{gray_value:02x}{gray_value:02x}"

        # Map importance to width (0.0=0.5, 1.0=3.0)
        edge_width = 0.5 + importance * 2.5

        if from_node_id in node_to_snake and to_node_id in node_to_snake:
            if node_to_snake[from_node_id] == node_to_snake[to_node_id]:
                # Same snake: use snake color with importance-based width
                snake_id = node_to_snake[from_node_id]
                snake_color = colors[snake_id % len(colors)]
                dot.edge(to_id, from_id, color=snake_color, penwidth=str(edge_width))
            else:
                # Different snakes: use importance-based gray color and width
                dot.edge(to_id, from_id, color=edge_color, penwidth=str(edge_width))
        else:
            # At least one node not in a snake: use importance-based gray color and width
            dot.edge(to_id, from_id, color=edge_color, penwidth=str(edge_width))

    # Render to SVG
    output_path_str = str(output_path.with_suffix(""))
    dot.render(output_path_str, cleanup=True)

    svg_path = Path(f"{output_path_str}.svg")

    # Generate HTML wrapper
    html_path = Path(f"{output_path_str}.html")
    _generate_html_wrapper(svg_path, html_path, graph_data, snakes, node_to_snake, edge_importance)

    print(f"\nSnake visualization saved to: {html_path}")
    print(f"Open it in your browser: file://{html_path.resolve()}")


def _generate_html_wrapper(
    svg_path: Path,
    html_path: Path,
    graph_data: dict,
    snakes: list[list[int]],
    node_to_snake: dict,
    edge_importance: dict[frozenset, float] | None = None,
) -> None:
    """Generate HTML file with interactive tooltips and snake legend.

    Args:
        svg_path: Path to SVG file
        html_path: Path to output HTML file
        graph_data: Graph data dict
        snakes: List of detected snakes
        node_to_snake: Mapping from node ID to snake ID
        edge_importance: Optional edge importance scores
    """
    # Read SVG content and extract dimensions
    with open(svg_path, encoding="utf-8") as f:
        svg_content = f.read()

    # Extract SVG viewBox to get actual dimensions
    import re as re_module

    viewbox_match = re_module.search(r'viewBox="([^"]+)"', svg_content)
    if viewbox_match:
        viewbox = viewbox_match.group(1).split()
        svg_width = float(viewbox[2])
        svg_height = float(viewbox[3])
    else:
        # Fallback: parse width/height attributes
        width_match = re_module.search(r'width="([\d.]+)pt"', svg_content)
        height_match = re_module.search(r'height="([\d.]+)pt"', svg_content)
        svg_width = float(width_match.group(1)) if width_match else 1000
        svg_height = float(height_match.group(1)) if height_match else 2000

    # Remove xlink:title attributes
    svg_content = re_module.sub(r'\s*xlink:title="[^"]*"', "", svg_content)

    # Build node data
    node_data = {}
    for node in graph_data["nodes"]:
        node_id = str(node["id"])
        node_info = {
            "id": node_id,
            "sentence_id": node["sentence_id"],
            "label": node["label"],
            "content": node["content"],
        }
        # Add snake info
        if node["id"] in node_to_snake:
            node_info["snake_id"] = node_to_snake[node["id"]]
        node_data[node_id] = node_info

    # Build snake legend
    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#FFA07A",
        "#98D8C8",
        "#F7DC6F",
        "#BB8FCE",
        "#85C1E2",
        "#F8B88B",
        "#A3E4D7",
    ]

    legend_items = []
    for i, snake in enumerate(snakes):
        color = colors[i % len(colors)]
        # Get first and last chunk labels
        first_node = next(n for n in graph_data["nodes"] if n["id"] == snake[0])
        last_node = next(n for n in graph_data["nodes"] if n["id"] == snake[-1])
        legend_items.append(
            f'<div class="legend-item">'
            f'<span class="legend-color" style="background-color: {color};"></span>'
            f'<span class="legend-text">Snake {i}: {first_node["label"]} → {last_node["label"]} ({len(snake)} nodes)</span>'
            f"</div>"
        )

    legend_html = "\n".join(legend_items)

    # Build edge importance legend
    edge_legend_html = ""
    if edge_importance:
        edge_legend_html = """
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                <h4 style="margin: 0 0 8px 0; font-size: 14px; color: #333;">Edge Importance</h4>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 12px; color: #666;">Low</span>
                    <div style="flex: 1; height: 8px; background: linear-gradient(to right, #CCCCCC, #000000); border-radius: 4px;"></div>
                    <span style="font-size: 12px; color: #666;">High</span>
                    <span style="margin-left: 10px; font-size: 12px; color: #888;">(shown by edge thickness and darkness)</span>
                </div>
            </div>
        """

    # Create HTML
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snake Detection - Knowledge Graph</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            overflow: auto;
        }}

        #container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}

        #legend {{
            background: white;
            padding: 15px 20px;
            border-bottom: 1px solid #ddd;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex-shrink: 0;
        }}

        #legend h3 {{
            margin: 0 0 10px 0;
            font-size: 16px;
            color: #333;
        }}

        .legend-item {{
            display: inline-block;
            margin: 4px 12px 4px 0;
            font-size: 13px;
        }}

        .legend-color {{
            display: inline-block;
            width: 16px;
            height: 16px;
            border-radius: 2px;
            margin-right: 6px;
            border: 2px solid rgba(0,0,0,0.3);
            vertical-align: middle;
        }}

        .legend-text {{
            color: #555;
            vertical-align: middle;
        }}

        #svg-container {{
            flex: 1;
            overflow: auto;
            background: white;
            padding: 20px;
            position: relative;
            overscroll-behavior: contain;
            min-height: 0;
        }}

        #svg-wrapper {{
            display: inline-block;
            min-width: fit-content;
            min-height: fit-content;
        }}

        svg {{
            display: block;
            max-width: none;
            max-height: none;
            width: auto;
            height: auto;
        }}

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

        .tooltip-snake {{
            color: #4ECDC4;
            font-weight: bold;
            font-size: 12px;
            margin-bottom: 4px;
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
        <div id="legend">
            <h3>🐍 Detected Snakes (Thematic Chains)</h3>
            {legend_html}
            {edge_legend_html}
        </div>
        <div id="svg-container">
            <div id="svg-wrapper">
                {svg_content}
            </div>
        </div>
    </div>
    <div id="tooltip"></div>

    <script>
        const nodeData = {json.dumps(node_data, ensure_ascii=False)};
        const tooltip = document.getElementById('tooltip');
        const svg = document.querySelector('svg');
        const nodes = svg.querySelectorAll('g.node');
        const svgContainer = document.getElementById('svg-container');

        // Prevent browser back navigation on horizontal swipe in SVG container
        let touchStartX = 0;
        let touchStartY = 0;

        svgContainer.addEventListener('touchstart', (e) => {{
            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;
        }}, {{ passive: true }});

        svgContainer.addEventListener('touchmove', (e) => {{
            const touchEndX = e.touches[0].clientX;
            const touchEndY = e.touches[0].clientY;
            const diffX = touchEndX - touchStartX;
            const diffY = touchEndY - touchStartY;

            // Only prevent if:
            // 1. Horizontal swipe is dominant (more than 2:1 ratio)
            // 2. Moving right (positive diffX) near left edge
            // 3. Container is scrolled to the left edge
            const isHorizontalDominant = Math.abs(diffX) > Math.abs(diffY) * 2;
            const isSwipingRight = diffX > 30;
            const isAtLeftEdge = svgContainer.scrollLeft <= 10;

            if (isHorizontalDominant && isSwipingRight && isAtLeftEdge) {{
                e.preventDefault();
            }}
        }}, {{ passive: false }});


        nodes.forEach(node => {{
            const title = node.querySelector('title');
            if (!title) return;

            const nodeId = title.textContent.trim();
            const data = nodeData[nodeId];
            if (!data) return;

            node.addEventListener('mouseenter', (e) => {{
                let snakeInfo = '';
                if (data.snake_id !== undefined) {{
                    snakeInfo = `<div class="tooltip-snake">🐍 Snake ${{data.snake_id}}</div>`;
                }}

                const tooltipHTML = `
                    ${{snakeInfo}}
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

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
