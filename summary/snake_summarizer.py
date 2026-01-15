"""Summarize detected snakes using LLM."""

from pathlib import Path

import networkx as nx

from summary.llm import LLM


class SnakeSummarizer:
    """Generate narrative summaries for detected thematic chains (snakes).

    Uses LLM to analyze the content of a snake and generate a concise
    narrative summary that captures the main theme and progression.
    """

    def __init__(self, llm: LLM, prompt_template_path: Path):
        """Initialize summarizer with LLM client and prompt template.

        Args:
            llm: LLM client for generating summaries
            prompt_template_path: Path to Jinja template for summarization prompt
        """
        self.llm = llm
        self.prompt_template_path = prompt_template_path

    def summarize_snake(self, snake_nodes: list[dict]) -> str:
        """Generate a narrative summary for a snake.

        Args:
            snake_nodes: List of node dicts with keys:
                - id: Node ID
                - label: Thematic label
                - content: Full content text
                - sentence_id: Position in original text
                (Should be sorted by sentence_id before passing)

        Returns:
            Summary text (2-4 sentences)
        """
        # Load and render prompt template
        system_prompt = self.llm.load_system_prompt(
            self.prompt_template_path,
            snake_nodes=snake_nodes,
        )

        # Call LLM with lower temperature for consistent summaries
        response = self.llm.request(
            system_prompt=system_prompt,
            user_message="请生成摘要。",
            temperature=0.4,
        )

        return response.strip()

    def summarize_all_snakes(
        self, snakes: list[list[int]], graph: nx.DiGraph
    ) -> list[dict]:
        """Generate summaries for all snakes.

        Args:
            snakes: List of snakes (each is a list of node IDs)
            graph: NetworkX graph with node attributes (id, label, content, sentence_id)

        Returns:
            List of summary dicts with:
                - snake_id: Snake index
                - size: Number of nodes
                - first_label: Label of first node
                - last_label: Label of last node
                - node_ids: List of node IDs
                - summary: Generated summary text
        """
        summaries = []

        for snake_id, snake in enumerate(snakes):
            # Extract node data from graph (sorted by sentence_id)
            snake_nodes = []
            for node_id in snake:
                node_data = graph.nodes[node_id]
                snake_nodes.append(
                    {
                        "id": node_id,
                        "label": node_data["label"],
                        "content": node_data["content"],
                        "sentence_id": node_data["sentence_id"],
                    }
                )

            # Sort by sentence_id for chronological order
            snake_nodes.sort(key=lambda n: n["sentence_id"])

            # Generate summary
            print(f"  Summarizing Snake {snake_id} ({len(snake_nodes)} nodes)...")
            summary_text = self.summarize_snake(snake_nodes)

            # Build result
            summaries.append(
                {
                    "snake_id": snake_id,
                    "size": len(snake_nodes),
                    "first_label": snake_nodes[0]["label"],
                    "last_label": snake_nodes[-1]["label"],
                    "node_ids": [n["id"] for n in snake_nodes],
                    "summary": summary_text,
                }
            )

        return summaries
