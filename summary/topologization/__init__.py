"""Topologization module - Cognitive chunk extraction and thematic chain detection.

This module provides tools for extracting cognitive chunks from text using LLM,
managing working memory with Wave Reflection algorithm, and detecting thematic
chains (snakes) in the resulting knowledge graph.

Main APIs:
---------
- TopologizationPipeline: Complete end-to-end pipeline
- KnowledgeGraphExtractor: Extract knowledge graph only
- ThematicChainAnalyzer: Detect and analyze thematic chains

Example:
--------
    from summary.topologization import TopologizationPipeline, PipelineConfig

    config = PipelineConfig(
        input_file=Path("input.txt"),
        output_dir=Path("output"),
        config_file=Path("format.json"),
    )
    pipeline = TopologizationPipeline(config)
    result = pipeline.run()
"""

from .core import (
    KnowledgeGraphExtractor,
    PipelineConfig,
    PipelineResult,
    ThematicChainAnalyzer,
    TopologizationPipeline,
)

__all__ = [
    "TopologizationPipeline",
    "KnowledgeGraphExtractor",
    "ThematicChainAnalyzer",
    "PipelineConfig",
    "PipelineResult",
]
