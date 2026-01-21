"""Fragment grouping using resource segmentation based on edge incision strength.

This module computes incision levels for each fragment based on external edge weights,
then uses resource-segmentation to group fragments within chapters.
"""

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from resource_segmentation import Resource, split


@dataclass
class FragmentInfo:
    """Information about a fragment including its incision levels."""

    chapter_id: int
    fragment_id: int
    token_count: int
    start_incision: float  # Raw incision value (to be normalized)
    end_incision: float  # Raw incision value (to be normalized)


@dataclass
class GroupInfo:
    """Information about a fragment group within a chapter."""

    chapter_id: int
    group_id: int
    fragment_ids: list[int]  # Fragment IDs in this group


def _get_fragment_token_count(workspace_path: Path, chapter_id: int, fragment_id: int) -> int:
    """Get total token count for a fragment by summing all sentence tokens.

    Args:
        workspace_path: Workspace path
        chapter_id: Chapter ID
        fragment_id: Fragment ID

    Returns:
        Total token count for the fragment
    """
    fragments_dir = workspace_path / "fragments"
    chapter_dir = fragments_dir / f"chapter-{chapter_id}"
    fragment_path = chapter_dir / f"fragment_{fragment_id}.json"

    with open(fragment_path, encoding="utf-8") as f:
        fragment_data = json.load(f)

    # Handle both old format (list) and new format (dict with "sentences")
    if isinstance(fragment_data, list):
        sentences = fragment_data
    else:
        sentences = fragment_data["sentences"]

    return sum(s["token_count"] for s in sentences)


def _compute_fragment_incisions(conn: sqlite3.Connection, workspace_path: Path) -> list[FragmentInfo]:
    """Compute incision levels for all fragments based on external edge weights.

    For each fragment:
    1. Find all chunks in the fragment
    2. Find all external edges (pointing to chunks in different fragments)
    3. Compute half-weight for each external edge: (chunk.weight / sum_external_edge_weights) * edge.weight
    4. Accumulate half-weights to start_incision (for edges pointing backward) or end_incision (forward)

    Args:
        conn: Database connection
        workspace_path: Workspace path

    Returns:
        List of FragmentInfo objects with raw incision values
    """
    cursor = conn.cursor()

    # Get all unique (chapter_id, fragment_id) pairs from chunks
    cursor.execute("""
        SELECT DISTINCT chapter_id, fragment_id
        FROM chunks
        ORDER BY chapter_id, fragment_id
    """)
    fragment_pairs = cursor.fetchall()

    fragment_infos = []

    for chapter_id, fragment_id in fragment_pairs:
        # Get token count for this fragment
        token_count = _get_fragment_token_count(workspace_path, chapter_id, fragment_id)

        # Get all chunks in this fragment
        cursor.execute(
            """
            SELECT id, weight
            FROM chunks
            WHERE chapter_id = ? AND fragment_id = ?
        """,
            (chapter_id, fragment_id),
        )
        chunks = cursor.fetchall()
        chunk_ids = [c[0] for c in chunks]
        chunk_weights = {c[0]: c[1] for c in chunks}

        if not chunk_ids:
            # No chunks in this fragment, use zero incisions
            fragment_infos.append(
                FragmentInfo(
                    chapter_id=chapter_id,
                    fragment_id=fragment_id,
                    token_count=token_count,
                    start_incision=0.0,
                    end_incision=0.0,
                )
            )
            continue

        start_incision = 0.0
        end_incision = 0.0

        # For each chunk, compute its contribution to incisions
        for chunk_id in chunk_ids:
            chunk_weight = chunk_weights[chunk_id]

            # Get all edges involving this chunk (treat as undirected)
            cursor.execute(
                """
                SELECT from_id, to_id, weight
                FROM knowledge_edges
                WHERE from_id = ? OR to_id = ?
            """,
                (chunk_id, chunk_id),
            )
            edges = cursor.fetchall()

            # Classify edges as internal or external
            external_edges = []
            for from_id, to_id, edge_weight in edges:
                other_chunk_id = to_id if from_id == chunk_id else from_id

                # Get fragment_id of the other chunk
                cursor.execute(
                    """
                    SELECT chapter_id, fragment_id
                    FROM chunks
                    WHERE id = ?
                """,
                    (other_chunk_id,),
                )
                result = cursor.fetchone()
                if result is None:
                    continue

                other_chapter_id, other_fragment_id = result

                # Check if external (different fragment)
                if (other_chapter_id, other_fragment_id) != (chapter_id, fragment_id):
                    external_edges.append(
                        {
                            "other_chapter_id": other_chapter_id,
                            "other_fragment_id": other_fragment_id,
                            "weight": edge_weight,
                        }
                    )

            if not external_edges:
                # No external edges for this chunk
                continue

            # Compute sum of external edge weights
            total_external_weight = sum(e["weight"] for e in external_edges)

            if total_external_weight == 0:
                continue

            # Compute half-weight for each external edge and accumulate to incisions
            for edge in external_edges:
                half_weight = (chunk_weight / total_external_weight) * edge["weight"]

                # Determine direction: forward (end_incision) or backward (start_incision)
                # Compare (chapter_id, fragment_id) tuples
                if (edge["other_chapter_id"], edge["other_fragment_id"]) < (chapter_id, fragment_id):
                    # Other fragment is before this one
                    start_incision += half_weight
                else:
                    # Other fragment is after this one
                    end_incision += half_weight

        fragment_infos.append(
            FragmentInfo(
                chapter_id=chapter_id,
                fragment_id=fragment_id,
                token_count=token_count,
                start_incision=start_incision,
                end_incision=end_incision,
            )
        )

    return fragment_infos


def _normalize_incisions(fragment_infos: list[FragmentInfo]) -> list[FragmentInfo]:
    """Normalize incision values to integer levels 1-10.

    Uses logarithmic scaling:
    1. Collect all non-zero incision values
    2. Sort and exclude top 2%
    3. Map remaining values using ln() to 1-10 range
    4. Top 2% get level 10

    Args:
        fragment_infos: List of FragmentInfo with raw incision values

    Returns:
        List of FragmentInfo with normalized incision levels (1-10 integers)
    """
    # Collect all non-zero incision values
    all_incisions = []
    for info in fragment_infos:
        if info.start_incision > 0:
            all_incisions.append(info.start_incision)
        if info.end_incision > 0:
            all_incisions.append(info.end_incision)

    if not all_incisions:
        # No incisions, everything stays 0
        return fragment_infos

    # Sort incisions
    all_incisions.sort()

    # Find threshold for top 2%
    threshold_index = int(len(all_incisions) * 0.98)
    if threshold_index >= len(all_incisions):
        threshold_index = len(all_incisions) - 1

    threshold = all_incisions[threshold_index]

    # Get non-top-2% values for ln scaling
    normal_values = [v for v in all_incisions if v < threshold]

    if not normal_values:
        # All values are in top 2%, just use 10
        normalized_infos = []
        for info in fragment_infos:
            normalized_infos.append(
                FragmentInfo(
                    chapter_id=info.chapter_id,
                    fragment_id=info.fragment_id,
                    token_count=info.token_count,
                    start_incision=10 if info.start_incision > 0 else 0,
                    end_incision=10 if info.end_incision > 0 else 0,
                )
            )
        return normalized_infos

    # Compute ln range for mapping
    min_val = min(normal_values)
    max_val = max(normal_values)

    # Avoid log(0) by using a small offset
    min_ln = math.log(min_val) if min_val > 0 else 0
    max_ln = math.log(max_val) if max_val > 0 else 0

    ln_range = max_ln - min_ln
    if ln_range == 0:
        ln_range = 1  # Avoid division by zero

    def normalize_value(val: float) -> int:
        """Normalize a single incision value to 1-10."""
        if val == 0:
            return 0
        if val >= threshold:
            return 10

        # Map using ln to 1-10
        val_ln = math.log(val) if val > 0 else min_ln
        normalized = 1 + ((val_ln - min_ln) / ln_range) * 9
        return max(1, min(10, int(round(normalized))))

    # Normalize all fragment incisions
    normalized_infos = []
    for info in fragment_infos:
        normalized_infos.append(
            FragmentInfo(
                chapter_id=info.chapter_id,
                fragment_id=info.fragment_id,
                token_count=info.token_count,
                start_incision=normalize_value(info.start_incision),
                end_incision=normalize_value(info.end_incision),
            )
        )

    return normalized_infos


def group_fragments_by_chapter(
    conn: sqlite3.Connection, workspace_path: Path, group_tokens_count: int
) -> list[GroupInfo]:
    """Group fragments within each chapter using resource segmentation.

    Args:
        conn: Database connection
        workspace_path: Workspace path
        group_tokens_count: Maximum token count per group (max_segment_count parameter)

    Returns:
        List of GroupInfo objects describing fragment groups
    """
    # Step 1: Compute raw incision values
    fragment_infos = _compute_fragment_incisions(conn, workspace_path)

    # Step 2: Normalize incisions to 1-10
    fragment_infos = _normalize_incisions(fragment_infos)

    # Step 3: Group fragments by chapter
    chapter_fragments: dict[int, list[FragmentInfo]] = {}
    for info in fragment_infos:
        if info.chapter_id not in chapter_fragments:
            chapter_fragments[info.chapter_id] = []
        chapter_fragments[info.chapter_id].append(info)

    # Step 4: Apply resource segmentation within each chapter
    all_groups = []

    for chapter_id, fragments in sorted(chapter_fragments.items()):
        # Sort fragments by fragment_id
        fragments.sort(key=lambda f: f.fragment_id)

        # Convert to Resource objects
        resources = []
        for info in fragments:
            resources.append(
                Resource(
                    count=info.token_count,
                    start_incision=int(info.start_incision),
                    end_incision=int(info.end_incision),
                    payload=info.fragment_id,  # Store fragment_id as payload
                )
            )

        # Apply resource segmentation (gap_rate=0, border_incision=0)
        groups = list(
            split(
                resources=iter(resources),  # Convert to iterator
                max_segment_count=group_tokens_count,
                border_incision=0,
                gap_rate=0.0,
                tail_rate=0.5,  # Doesn't matter since gap_rate=0
            )
        )

        # Extract fragment IDs from each group (only body, since gap_rate=0)
        for group_id, group in enumerate(groups):
            fragment_ids = [
                resource.payload for segment in group.body for resource in segment.resources  # type: ignore
            ]
            all_groups.append(
                GroupInfo(
                    chapter_id=chapter_id,
                    group_id=group_id,
                    fragment_ids=fragment_ids,
                )
            )

    return all_groups
