"""Topologization module - Workspace-based cognitive chunk extraction.

This module provides tools for extracting cognitive chunks from text using LLM,
managing working memory with Wave Reflection algorithm, and detecting thematic
chains (snakes) in the resulting knowledge graph.

The results are stored in a workspace with persistent storage (SQLite + fragments).

Main APIs:
---------
- topologize(): Execute pipeline and create workspace
- Topologization: Access results from workspace
- TopologizationConfig: Configuration for pipeline

Example:
--------
    from summary.topologization import topologize, TopologizationConfig
    from summary.llm import LLM
    from summary.template import create_env

    llm = LLM(config_path=Path("format.json"))
    jinja_env = create_env(data_dir)

    config = TopologizationConfig(
        extraction_prompt_file=data_dir / "extraction_prompt.jinja",
        snake_summary_prompt_file=data_dir / "snake_summary_prompt.jinja",
    )

    topo = topologize(
        input_file=Path("input.txt"),
        workspace_path=Path("workspace"),
        config=config,
        llm=llm,
        jinja_env=jinja_env,
    )

    # Access results
    for chunk in topo.knowledge_graph:
        print(chunk.label, chunk.content)
"""

from .api import Chunk, Snake, Topologization
from .storage import SentenceId
from .topologize import TopologizationConfig, topologize

__all__ = [
    # Main API
    "topologize",
    "Topologization",
    "TopologizationConfig",
    # Data types
    "Chunk",
    "Snake",
    "SentenceId",
]
