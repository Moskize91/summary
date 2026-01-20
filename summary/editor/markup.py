"""XML markup formatting for clue chunks.

Formats clue chunks as book-like text with XML-style markup for AI consumption.
Handles fragment loading, chunk overlap merging, and non-continuous chunk splitting.
"""

import json

from ..topologization.api import Topologization
from ..topologization.fragment import FragmentReader, SentenceId


def format_clue_as_book(chunks: list, topologization: Topologization, wrap_high_retention: bool = False) -> str:
    """Format clue chunks as book-like text with XML markup.

    Renders all fragments involved in the clue as continuous narrative text,
    with chunk boundaries marked using XML-style tags containing metadata.

    For fragments that are skipped between clue fragments, inserts their summaries
    as natural transition paragraphs.

    Args:
        chunks: List of Chunk objects
        topologization: Topologization object
        wrap_high_retention: If True, wrap high-retention chunks (verbatim/detailed) in <chunk> tags with source text

    Returns:
        Book-like text with <chunk> markup and fragment summaries

    Example output:
        ```
        Regular unmarked text from fragment...

        <chunk id="123" label="Key event" retention="detailed" importance="critical">
        Marked chunk text...
        </chunk>

        [Fragment summary for skipped fragments 2-3]

        More unmarked text...
        ```
    """
    # Step 1: Collect all fragments involved
    fragment_ids = sorted(set(sid[0] for chunk in chunks for sid in chunk.sentence_ids))

    if not fragment_ids:
        return ""

    # Step 2: Load all sentences from these fragments
    fragments_dir = topologization.workspace_path / "fragments"
    fragment_sentences: dict[int, list[tuple[int, str]]] = {}  # fragment_id -> [(sentence_index, text), ...]
    fragment_reader = FragmentReader(topologization.workspace_path)

    for frag_id in fragment_ids:
        fragment_path = fragments_dir / f"fragment_{frag_id}.json"
        with open(fragment_path, encoding="utf-8") as f:
            fragment_data = json.load(f)

        # Handle both old format (list) and new format (dict with "sentences")
        if isinstance(fragment_data, list):
            fragment_sentences[frag_id] = [(i, s["text"]) for i, s in enumerate(fragment_data)]
        else:
            fragment_sentences[frag_id] = [(i, s["text"]) for i, s in enumerate(fragment_data["sentences"])]

    # Step 3: Build chunk coverage map - handle overlaps
    # chunk_coverage: dict mapping (fragment_id, sentence_index) -> list of Chunk objects
    chunk_coverage: dict[SentenceId, list] = {}
    for chunk in chunks:
        for sid in chunk.sentence_ids:
            if sid not in chunk_coverage:
                chunk_coverage[sid] = []
            chunk_coverage[sid].append(chunk)

    # Step 4: Render all fragments with markup, inserting summaries for skipped fragments
    result_parts = []
    for i, frag_id in enumerate(fragment_ids):
        # Check if there are skipped fragments before this one
        if i > 0:
            prev_frag_id = fragment_ids[i - 1]
            # If there's a gap (skipped fragments), insert their summaries
            if frag_id > prev_frag_id + 1:
                skipped_summaries = []
                for skipped_id in range(prev_frag_id + 1, frag_id):
                    summary = fragment_reader.get_summary(skipped_id)
                    if summary:  # Only add non-empty summaries
                        skipped_summaries.append(summary)

                # Combine all skipped summaries into one natural paragraph
                if skipped_summaries:
                    result_parts.append(" ".join(skipped_summaries))

        # Render the current fragment
        marked_text = _build_fragment_markup(frag_id, fragment_sentences[frag_id], chunk_coverage, wrap_high_retention)
        result_parts.append(marked_text)

    return "\n\n".join(result_parts)


def _build_fragment_markup(
    frag_id: int,
    sentences: list[tuple[int, str]],
    chunk_coverage: dict[SentenceId, list],
    wrap_high_retention: bool = False,
) -> str:
    """Build marked-up text for one fragment.

    Args:
        frag_id: Fragment ID
        sentences: List of (sentence_index, text) tuples
        chunk_coverage: Map from sentence_id to list of Chunk objects
        wrap_high_retention: If True, wrap high-retention chunks in <chunk> tags

    Returns:
        Fragment text with <chunk> markup
    """
    # Group consecutive sentences into ranges
    # Each range is either: unmarked text OR chunk-marked text
    sentence_ranges: list[tuple[int, int, dict | None]] = []  # (start_idx, end_idx, chunk_attrs or None)

    i = 0
    while i < len(sentences):
        sent_idx, sent_text = sentences[i]
        sid = (frag_id, sent_idx)

        if sid in chunk_coverage:
            # Start of a chunk range
            chunk_attrs = _merge_chunk_attributes(chunk_coverage[sid])
            start_idx = i

            # Find consecutive sentences in same chunk(s)
            j = i + 1
            while j < len(sentences):
                next_sid = (frag_id, sentences[j][0])
                if next_sid not in chunk_coverage:
                    break
                # Check if same chunk coverage (check if label matches)
                if next_sid in chunk_coverage:
                    # NOTE: For simplicity, we create separate ranges if chunk coverage changes
                    # This handles non-continuous chunks automatically
                    next_attrs = _merge_chunk_attributes(chunk_coverage[next_sid])
                    if next_attrs["label"] != chunk_attrs["label"]:
                        break
                j += 1

            sentence_ranges.append((start_idx, j, chunk_attrs))
            i = j
        else:
            # Unmarked sentence
            sentence_ranges.append((i, i + 1, None))
            i += 1

    # Render text with markup
    parts = []
    for start_idx, end_idx, chunk_attrs in sentence_ranges:
        text_segment = " ".join(sentences[j][1] for j in range(start_idx, end_idx))

        if chunk_attrs:
            # Check if this chunk needs wrapping (for compressor view only)
            retention = chunk_attrs.get("retention")
            should_wrap = wrap_high_retention and retention in ("verbatim", "detailed")

            if should_wrap:
                # Wrap with <chunk retention="XXX"> for high-retention chunks
                parts.append(f'<chunk retention="{retention}">{text_segment}</chunk>')
            else:
                # Normal case: no wrapping, just plain text
                parts.append(text_segment)
        else:
            parts.append(text_segment)

    return " ".join(parts)


def _merge_chunk_attributes(chunks_at_position: list) -> dict:
    """Merge attributes when multiple chunks cover same sentence.

    Rules:
    - retention: verbatim > detailed > focused > relevant > None
    - importance: critical > important > helpful > None
    - label: Take from highest-priority chunk (retention first, then importance)

    Args:
        chunks_at_position: List of Chunk objects covering this sentence

    Returns:
        Merged attributes dict with keys: label, retention, importance
    """
    retention_order = {"verbatim": 4, "detailed": 3, "focused": 2, "relevant": 1, None: 0}
    importance_order = {"critical": 3, "important": 2, "helpful": 1, None: 0}

    # Find chunk with highest retention
    best_retention_chunk = max(chunks_at_position, key=lambda c: retention_order.get(c.retention, 0))
    retention = best_retention_chunk.retention

    # Find chunk with highest importance
    best_importance_chunk = max(chunks_at_position, key=lambda c: importance_order.get(c.importance, 0))
    importance = best_importance_chunk.importance

    # Use label from the chunk with higher overall priority (retention first, then importance)
    priority_chunk = max(
        chunks_at_position,
        key=lambda c: (retention_order.get(c.retention, 0), importance_order.get(c.importance, 0)),
    )
    label = priority_chunk.label

    return {
        "label": label,
        "retention": retention,
        "importance": importance,
    }
