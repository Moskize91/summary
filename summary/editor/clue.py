"""Clue extraction and merging for editor stage.

Converts Snake structures from topologization into Clues,
merging low-weight snakes to reduce reviewer count.
"""

from dataclasses import dataclass

from ..topologization.api import Chunk, Snake, Topologization


@dataclass
class Clue:
    """A narrative clue in the editor stage.

    Represents a coherent narrative thread, either converted from a single Snake
    or merged from multiple low-weight Snakes.

    Attributes:
        clue_id: Unique clue ID (original snake_id or -1/-2/... for merged clues)
        weight: Normalized weight (sum = 1.0 across all clues)
        label: Human-readable label (e.g., "first_label → last_label")
        chunks: All chunks in this clue, sorted by sentence_id
        is_merged: Whether this clue was created by merging multiple snakes
        source_snake_ids: List of original snake IDs (single element if not merged)
    """

    clue_id: int
    weight: float
    label: str
    chunks: list[Chunk]
    is_merged: bool
    source_snake_ids: list[int]


def extract_clues_from_topologization(
    topologization: Topologization,
    max_clues: int = 10,
) -> list[Clue]:
    """Extract clues from topologization, merging low-weight snakes.

    Strategy:
    1. Sort all snakes by weight (descending)
    2. Keep top (max_clues - 1) snakes as individual Clues
    3. Merge remaining snakes into one Clue
    4. Re-normalize all weights to sum to 1.0

    Args:
        topologization: Topologization object with snake graph
        max_clues: Maximum number of clues to generate (default: 10)

    Returns:
        List of Clue objects, sorted by weight (descending)
    """
    # Get all snakes and sort by weight (descending)
    snakes = list(topologization.snake_graph)
    snakes.sort(key=lambda s: s.weight, reverse=True)

    if not snakes:
        return []

    # If total snakes <= max_clues, no merging needed
    if len(snakes) <= max_clues:
        clues = [_convert_snake_to_clue(s, topologization) for s in snakes]
        # Normalize weights
        total_weight = sum(c.weight for c in clues)
        if total_weight > 0:
            for clue in clues:
                clue.weight = clue.weight / total_weight
        return clues

    # Split into high-weight and low-weight groups
    # Keep top (max_clues - 1) snakes, merge the rest
    high_weight_snakes = snakes[: max_clues - 1]
    low_weight_snakes = snakes[max_clues - 1 :]

    # Convert high-weight snakes to individual clues
    clues = [_convert_snake_to_clue(s, topologization) for s in high_weight_snakes]

    # Merge low-weight snakes into one clue (if any)
    if low_weight_snakes:
        merged_clue = _merge_snakes_into_clue(low_weight_snakes, topologization)
        clues.append(merged_clue)

    # Re-normalize weights
    clue_total_weight = sum(c.weight for c in clues)
    if clue_total_weight > 0:
        for clue in clues:
            clue.weight = clue.weight / clue_total_weight

    return clues


def _convert_snake_to_clue(snake: Snake, topologization: Topologization) -> Clue:
    """Convert a single Snake into a Clue.

    Args:
        snake: Snake object from topologization
        topologization: Topologization object for loading chunks

    Returns:
        Clue object
    """
    chunks = snake.get_chunks()

    return Clue(
        clue_id=snake.snake_id,
        weight=snake.weight,
        label=f"{snake.first_label} → {snake.last_label}",
        chunks=chunks,
        is_merged=False,
        source_snake_ids=[snake.snake_id],
    )


def _merge_snakes_into_clue(
    snakes: list[Snake],
    topologization: Topologization,
    merged_clue_id: int = -1,
) -> Clue:
    """Merge multiple low-weight snakes into a single Clue.

    Args:
        snakes: List of Snake objects to merge
        topologization: Topologization object for loading chunks
        merged_clue_id: ID for merged clue (default: -1)

    Returns:
        Merged Clue object
    """
    # Collect all chunks from all snakes
    all_chunks = []
    for snake in snakes:
        all_chunks.extend(snake.get_chunks())

    # Sort chunks by sentence_id to maintain text order
    # sentence_id is (fragment_id, sentence_index)
    all_chunks.sort(key=lambda c: c.sentence_id)

    # Calculate total weight
    total_weight = sum(s.weight for s in snakes)

    # Create label
    label = f"综合次要线索 ({len(snakes)}条)"

    # Collect source snake IDs
    source_ids = [s.snake_id for s in snakes]

    return Clue(
        clue_id=merged_clue_id,
        weight=total_weight,
        label=label,
        chunks=all_chunks,
        is_merged=True,
        source_snake_ids=source_ids,
    )
