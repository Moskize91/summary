"""Visualize snake-level graph with aggregated nodes."""

import json
from pathlib import Path

import networkx as nx
from graphviz import Digraph


def visualize_snake_graph(
    snake_graph: nx.DiGraph,
    output_path: Path,
    snake_summaries: list[dict],
) -> None:
    """Generate visualization with one node per snake.

    Args:
        snake_graph: NetworkX graph where nodes are snake IDs
        output_path: Output file path (without extension)
        snake_summaries: List of summary dicts with snake_id, summary, etc.
    """
    # Create graphviz digraph
    dot = Digraph(comment="Snake-Level Knowledge Graph", format="svg")

    # Configure graph attributes - BT (Bottom to Top) so earlier snakes appear at top
    dot.attr(rankdir="BT")
    dot.attr(splines="ortho")
    dot.attr(nodesep="2.0")
    dot.attr(ranksep="1.5")
    dot.attr(center="true")
    dot.attr(margin="0.5")

    # Configure default node attributes (larger for aggregated nodes)
    dot.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fontname="Arial",
        fontsize="14",
        margin="0.4,0.3",
    )

    # Configure default edge attributes
    dot.attr("edge", color="#848484", arrowsize="0.8")

    # Same color palette as node-level visualization
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

    # Build snake_id -> summary mapping
    summary_map = {s["snake_id"]: s for s in snake_summaries}

    # Add snake nodes
    for snake_id in snake_graph.nodes():
        node_data = snake_graph.nodes[snake_id]
        summary_data = summary_map[snake_id]

        # Node label
        label = (
            f"Snake {snake_id}\\n{node_data['first_label']} → {node_data['last_label']}\\n({node_data['size']} nodes)"
        )

        # Tooltip with summary and node IDs
        node_ids_str = ", ".join(str(nid) for nid in node_data["node_ids"])
        tooltip = (
            f"Snake {snake_id}\\n"
            f"Size: {node_data['size']} nodes\\n"
            f"Node IDs: {node_ids_str}\\n\\n"
            f"Summary:\\n{summary_data['summary']}"
        )

        # Color from palette
        color = colors[snake_id % len(colors)]

        # Add node with styling
        dot.node(
            str(snake_id),
            label=label,
            tooltip=tooltip,
            fillcolor=color + "80",  # Semi-transparent
            color=color,  # Border
            penwidth="3",
        )

    # Add edges between snakes
    for edge in snake_graph.edges():
        snake_from, snake_to = edge
        edge_data = snake_graph.edges[edge]

        importance = edge_data["importance"]
        internal_edge_count = edge_data["internal_edge_count"]

        # Map importance to color (0.0=light gray, 1.0=black)
        gray_value = int(204 * (1.0 - importance))
        edge_color = f"#{gray_value:02x}{gray_value:02x}{gray_value:02x}"

        # Map importance + count to width
        # Base width from importance, bonus from count
        base_width = 0.5 + importance * 2.5  # [0.5, 3.0]
        count_bonus = min(internal_edge_count * 0.2, 1.5)  # Up to +1.5
        edge_width = base_width + count_bonus

        # Add edge
        dot.edge(
            str(snake_from),
            str(snake_to),
            color=edge_color,
            penwidth=str(edge_width),
        )

    # Render to SVG
    output_path_str = str(output_path.with_suffix(""))
    dot.render(output_path_str, cleanup=True)

    svg_path = Path(f"{output_path_str}.svg")

    # Generate HTML wrapper
    html_path = Path(f"{output_path_str}.html")
    _generate_html_wrapper(svg_path, html_path, snake_graph, snake_summaries)

    print(f"\nSnake graph visualization saved to: {html_path}")
    print(f"Open it in your browser: file://{html_path.resolve()}")


def _generate_html_wrapper(
    svg_path: Path,
    html_path: Path,
    snake_graph: nx.DiGraph,
    snake_summaries: list[dict],
) -> None:
    """Generate HTML file with interactive tooltips and legend.

    Args:
        svg_path: Path to SVG file
        html_path: Path to output HTML file
        snake_graph: Snake-level graph
        snake_summaries: List of summary dicts
    """
    # Read SVG content
    with open(svg_path, encoding="utf-8") as f:
        svg_content = f.read()

    # Extract SVG viewBox
    import re

    viewbox_match = re.search(r'viewBox="([^"]+)"', svg_content)
    if viewbox_match:
        viewbox = viewbox_match.group(1).split()
        svg_width = float(viewbox[2])
        svg_height = float(viewbox[3])
    else:
        width_match = re.search(r'width="([\d.]+)pt"', svg_content)
        height_match = re.search(r'height="([\d.]+)pt"', svg_content)
        svg_width = float(width_match.group(1)) if width_match else 1000
        svg_height = float(height_match.group(1)) if height_match else 1000

    # Remove xlink:title attributes
    svg_content = re.sub(r'\s*xlink:title="[^"]*"', "", svg_content)

    # Build snake data for JavaScript
    summary_map = {s["snake_id"]: s for s in snake_summaries}
    snake_data = {}
    for snake_id in snake_graph.nodes():
        node_data = snake_graph.nodes[snake_id]
        summary_data = summary_map[snake_id]

        snake_data[str(snake_id)] = {
            "snake_id": snake_id,
            "size": node_data["size"],
            "first_label": node_data["first_label"],
            "last_label": node_data["last_label"],
            "node_ids": node_data["node_ids"],
            "summary": summary_data["summary"],
            "chunks": summary_data.get("chunks", []),
        }

    # Build legend
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
    for snake_id in sorted(snake_graph.nodes()):
        color = colors[snake_id % len(colors)]
        node_data = snake_graph.nodes[snake_id]
        summary_data = summary_map[snake_id]

        legend_items.append(
            f'<div class="legend-item">'
            f'<span class="legend-color" style="background-color: {color};"></span>'
            f'<span class="legend-text">Snake {snake_id}: {node_data["first_label"]} → {node_data["last_label"]} ({node_data["size"]} nodes)</span>'
            f"</div>"
        )

    legend_html = "\n".join(legend_items)

    # Create HTML
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snake-Level Knowledge Graph</title>
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
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 14px;
            line-height: 1.6;
            max-width: 700px;
            max-height: 80vh;
            overflow-y: auto;
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

        .tooltip-header {{
            font-weight: bold;
            color: #FFF700;
            font-size: 15px;
            margin-bottom: 6px;
        }}

        .tooltip-meta {{
            color: #aaa;
            font-size: 12px;
            margin-bottom: 8px;
        }}

        .tooltip-summary {{
            border-top: 1px solid #555;
            padding-top: 8px;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="legend">
            <h3>🐍 Snake-Level Graph (Aggregated View)</h3>
            {legend_html}
        </div>
        <div id="svg-container">
            <div id="svg-wrapper">
                {svg_content}
            </div>
        </div>
    </div>
    <div id="tooltip"></div>

    <script>
        const snakeData = {json.dumps(snake_data, ensure_ascii=False)};
        const tooltip = document.getElementById('tooltip');
        const svg = document.querySelector('svg');
        const nodes = svg.querySelectorAll('g.node');
        const svgContainer = document.getElementById('svg-container');

        // Prevent browser back navigation on horizontal swipe
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

            const snakeId = title.textContent.trim();
            const data = snakeData[snakeId];
            if (!data) return;

            node.addEventListener('mouseenter', (e) => {{
                const nodeIdsStr = data.node_ids.join(', ');

                // Build chunks HTML
                let chunksHTML = '';
                if (data.chunks && data.chunks.length > 0) {{
                    chunksHTML = '<div class="tooltip-chunks">';
                    chunksHTML += '<div style="font-weight: bold; margin-top: 8px; margin-bottom: 4px;">Chunks:</div>';
                    data.chunks.forEach((chunk, idx) => {{
                        chunksHTML += `<div style="margin-bottom: 6px; padding-left: 8px; border-left: 2px solid #666;">`;
                        chunksHTML += `<div style="color: #FFF700; font-size: 13px;">${{idx + 1}}. [${{chunk.label}}] (ID: ${{chunk.id}})</div>`;
                        chunksHTML += `<div style="color: #ccc; font-size: 12px; margin-top: 2px;">${{chunk.content}}</div>`;
                        chunksHTML += `</div>`;
                    }});
                    chunksHTML += '</div>';
                }}

                const tooltipHTML = `
                    <div class="tooltip-header">Snake ${{data.snake_id}}: ${{data.first_label}} → ${{data.last_label}}</div>
                    <div class="tooltip-meta">Size: ${{data.size}} nodes | Node IDs: ${{nodeIdsStr}}</div>
                    <div class="tooltip-summary">${{data.summary}}</div>
                    ${{chunksHTML}}
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
