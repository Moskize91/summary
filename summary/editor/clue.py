"""Clue extraction and merging for editor stage.

Converts Snake structures from topologization into Clues,
merging low-weight snakes to reduce reviewer count.
"""

from dataclasses import dataclass

from ..topologization import Chunk, Snake, Topologization


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
    chapter_id: int | None = None,
    group_id: int | None = None,
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
        chapter_id: If provided, only extract snakes from this chapter (requires group_id)
        group_id: If provided, only extract snakes from this group (requires chapter_id)

    Returns:
        List of Clue objects, sorted by weight (descending)
    """
    # Get snakes from specified group or all groups
    all_snakes = []
    if chapter_id is not None and group_id is not None:
        # Extract snakes from specific group
        snake_graph = topologization.get_snake_graph(chapter_id, group_id)
        all_snakes.extend(list(snake_graph))
    else:
        # Extract snakes from all chapters and groups
        for cid in topologization.get_all_chapter_ids():
            for gid in topologization.get_group_ids_for_chapter(cid):
                snake_graph = topologization.get_snake_graph(cid, gid)
                all_snakes.extend(list(snake_graph))

    if not all_snakes:
        return []

    # Convert all snakes to individual clues
    clues = [_convert_snake_to_clue(s, topologization) for s in all_snakes]

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
        # Sort by weight descending to determine ranking
        clues.sort(key=lambda c: c.weight, reverse=True)

        # Determine candidate region: clues ranked beyond max_clues * 0.75
        cutoff_rank = int(max_clues * 0.75)
        candidates = clues[cutoff_rank:]  # Long-tail region

        # If no candidates (shouldn't happen given loop condition), break
        if len(candidates) < 2:
            break

        # Find the pair with maximum fragment overlap (maximum reduction)
        best_pair = None
        best_reduction = -1

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                clue1 = candidates[i]
                clue2 = candidates[j]

                # Calculate fragment reduction
                reduction = _calculate_fragment_reduction(clue1, clue2)

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_pair = (clue1, clue2)

        # Merge the best pair
        if best_pair is None:
            # Fallback: if no valid pair found, take the two smallest by weight
            clues.sort(key=lambda c: c.weight)
            clue1, clue2 = clues[0], clues[1]
        else:
            clue1, clue2 = best_pair

        merged_clue = _merge_two_clues(clue1, clue2, merged_clue_id_counter)
        merged_clue_id_counter -= 1

        # Remove the two original clues and add the merged one
        clues.remove(clue1)
        clues.remove(clue2)
        clues.append(merged_clue)

    # Re-normalize weights
    total_weight = sum(c.weight for c in clues)
    if total_weight > 0:
        for clue in clues:
            clue.weight = clue.weight / total_weight

    # Sort by weight descending for final output
    clues.sort(key=lambda c: c.weight, reverse=True)

    return clues


def _calculate_fragment_reduction(clue1: Clue, clue2: Clue) -> int:
    """Calculate fragment reduction when merging two clues.

    Fragment reduction = (clue1 fragments + clue2 fragments) - merged fragments
    Higher reduction means more overlap, better candidates for merging.

    Args:
        clue1: First clue
        clue2: Second clue

    Returns:
        Number of fragments reduced (overlap count)
    """
    # Extract all fragment IDs from clue1's chunks
    fragments1 = set()
    for chunk in clue1.chunks:
        for sentence_id in chunk.sentence_ids:
            fragment_id = sentence_id[0]  # sentence_id is (fragment_id, sentence_index)
            fragments1.add(fragment_id)

    # Extract all fragment IDs from clue2's chunks
    fragments2 = set()
    for chunk in clue2.chunks:
        for sentence_id in chunk.sentence_ids:
            fragment_id = sentence_id[0]
            fragments2.add(fragment_id)

    # Merged fragments (union)
    merged_fragments = fragments1 | fragments2

    # Reduction = sum of individual - merged
    reduction = len(fragments1) + len(fragments2) - len(merged_fragments)

    return reduction


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
