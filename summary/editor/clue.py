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
    1. Start with all snakes as individual clues
    2. Repeatedly merge the two lowest-weight clues until count <= max_clues
    3. Re-normalize all weights to sum to 1.0

    This creates a hierarchical merging where the smallest clues are combined first,
    ensuring that only the true tail gets merged while significant clues remain independent.

    Args:
        topologization: Topologization object with snake graph
        max_clues: Maximum number of clues to generate (default: 10)

    Returns:
        List of Clue objects, sorted by weight (descending)
    """
    # Get all snakes and convert to initial clues
    snakes = list(topologization.snake_graph)

    if not snakes:
        return []

    # Convert all snakes to individual clues
    clues = [_convert_snake_to_clue(s, topologization) for s in snakes]

    # If already within limit, no merging needed
    if len(clues) <= max_clues:
        # Normalize weights
        total_weight = sum(c.weight for c in clues)
        if total_weight > 0:
            for clue in clues:
                clue.weight = clue.weight / total_weight
        # Sort by weight descending
        clues.sort(key=lambda c: c.weight, reverse=True)
        return clues

    # Merge smallest clues until we reach max_clues
    merged_clue_id_counter = -1
    while len(clues) > max_clues:
        # Sort by weight ascending to find the two smallest
        clues.sort(key=lambda c: c.weight)

        # Take the two smallest clues
        clue1 = clues[0]
        clue2 = clues[1]

        # Merge them
        merged_clue = _merge_two_clues(clue1, clue2, merged_clue_id_counter)
        merged_clue_id_counter -= 1

        # Remove the two original clues and add the merged one
        clues = clues[2:]  # Remove first two
        clues.append(merged_clue)

    # Re-normalize weights
    total_weight = sum(c.weight for c in clues)
    if total_weight > 0:
        for clue in clues:
            clue.weight = clue.weight / total_weight

    # Sort by weight descending for final output
    clues.sort(key=lambda c: c.weight, reverse=True)

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


def _merge_two_clues(clue1: Clue, clue2: Clue, merged_clue_id: int) -> Clue:
    """Merge two clues into one.

    Used in hierarchical merging - either clue may already be a merged clue.

    Args:
        clue1: First clue to merge
        clue2: Second clue to merge
        merged_clue_id: ID for the new merged clue

    Returns:
        Merged Clue object
    """
    # Combine all chunks
    all_chunks = list(clue1.chunks) + list(clue2.chunks)

    # Sort chunks by sentence_id to maintain text order
    all_chunks.sort(key=lambda c: c.sentence_id)

    # Calculate total weight
    total_weight = clue1.weight + clue2.weight

    # Collect source snake IDs
    source_ids = list(clue1.source_snake_ids) + list(clue2.source_snake_ids)

    # Create label
    label = f"综合次要线索 ({len(source_ids)}条)"

    return Clue(
        clue_id=merged_clue_id,
        weight=total_weight,
        label=label,
        chunks=all_chunks,
        is_merged=True,
        source_snake_ids=source_ids,
    )
