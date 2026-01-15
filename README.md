# Summary

A knowledge graph extraction and visualization tool with snake detection.

## Quick Start

### Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
python test.py
```

### Usage

1. **Extract knowledge graph from text:**
   ```bash
   python scripts/main.py
   ```
   - Processes text from `input/` directory
   - Generates `output/knowledge_graph.json`

2. **Visualize with snake detection:**
   ```bash
   python scripts/visualize.py
   ```
   - Detects thematic chains ("snakes") in the knowledge graph
   - Generates color-coded visualization at `output/knowledge_graph.html`
   - Each snake is shown in a different color
   - Open the HTML file in your browser to explore

## Features

- **Knowledge Graph Extraction**: Extract entities and relationships from text using LLM
- **Wave Reflection Algorithm**: Smart working memory selection with generation decay
- **Snake Detection**: Discover thematic chains using ink diffusion clustering
  - Hierarchical clustering with average linkage
  - Automatic detection of narrative structures
  - No need to specify number of themes upfront
- **Interactive Visualization**: Color-coded graphs with tooltips and legends
- Modern PEP 621 compliant `pyproject.toml`
- Pre-configured development tools:
  - `pytest` for testing
  - `ruff` for linting and formatting
  - `pylint` for additional linting
  - `pyright` for type checking

## Snake Detection

"Snakes" are thematic chains - groups of chunks that share similar topological patterns in the knowledge graph. The algorithm:

1. Computes "ink diffusion" signatures for each node (multi-hop neighborhood)
2. Converts signatures to numerical fingerprints
3. Uses hierarchical clustering to group similar nodes
4. Identifies continuous narrative threads

Example output:
- **Snake 0** (4 nodes): 元朝压迫线索
- **Snake 4** (10 nodes): 从苦难到造反 ⭐ Main storyline

See `output/snakes.json` for detailed results.

## Project Structure

```
.
├── summary/                # Source code
│   ├── cognitive_chunk.py  # Chunk data structure
│   ├── wave_reflection.py  # Working memory algorithm
│   └── ...
├── dev/                    # Development tools
│   ├── snake_detector.py   # Snake detection algorithm
│   ├── visualize_snakes.py # Visualization with snakes
│   └── visualizer.py       # DEPRECATED: Basic visualization
├── scripts/                # Runnable scripts
│   ├── main.py            # Main extraction pipeline
│   └── visualize.py       # Visualize with snake detection
├── tests/                  # Test files
├── input/                  # Input text files
├── output/                 # Generated files
│   ├── knowledge_graph.json
│   ├── knowledge_graph.html
│   └── snakes.json
└── README.md
```

## Development Tools

Run linting:
```bash
ruff check .
pylint ./summary/**/*.py ./tests/**/*.py
```

Run type checking:
```bash
pyright summary tests
```

Run formatting:
```bash
ruff format .
```

Run tests:
```bash
pytest tests/ -v
# or
python test.py
```

## License

MIT License - see [LICENSE](LICENSE) file for details.