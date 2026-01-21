"""XML markup formatting for clue chunks.

Formats clue chunks as book-like text with XML-style markup for AI consumption.
Handles fragment loading, chunk overlap merging, and non-continuous chunk splitting.
"""

from ..topologization.api import Topologization
from ..topologization.fragment import FragmentReader, SentenceId


def format_clue_as_book(
    chunks: list,
    topologization: Topologization,
    wrap_high_retention: bool = False,
    full_markup: bool = False,
) -> str:
    """Format clue chunks as book-like text with XML markup.

    Renders all fragments involved in the clue as continuous narrative text,
    with chunk boundaries marked using XML-style tags containing metadata.

    For fragments that are skipped between clue fragments, inserts their summaries
    as natural transition paragraphs.

    Args:
        chunks: List of Chunk objects
        topologization: Topologization object
        wrap_high_retention: If True, wrap only high-retention chunks (verbatim/detailed)
            with simplified tags (for compressor)
        full_markup: If True, wrap ALL chunks with full XML attributes including label,
            retention, importance (for reviewer generator)

    Returns:
        Book-like text with <chunk> markup and fragment summaries

    Example output (full_markup=True):
        ```
        Regular unmarked text from fragment...

        <chunk label="Key event" retention="detailed" importance="critical">
        Marked chunk text...
        </chunk>

        [Fragment summary for skipped fragments 2-3]

        More unmarked text...
        ```
    """
    # Step 1: Collect all (chapter_id, fragment_id) pairs involved
    chapter_fragment_pairs = sorted(set((sid[0], sid[1]) for chunk in chunks for sid in chunk.sentence_ids))

    if not chapter_fragment_pairs:
        return ""

    # Step 2: Load all sentences from these fragments using FragmentReader
    fragment_sentences: dict[tuple[int, int], list[tuple[int, str]]] = {}
    fragment_reader = FragmentReader(topologization.workspace_path)

    for chapter_id, frag_id in chapter_fragment_pairs:
        # Get all sentences for this fragment
        sentences = []
        sentence_index = 0
        while True:
            try:
                sentence_text = fragment_reader.get_sentence((chapter_id, frag_id, sentence_index))
                sentences.append((sentence_index, sentence_text))
                sentence_index += 1
            except (IndexError, KeyError):
                # No more sentences in this fragment
                break
        fragment_sentences[(chapter_id, frag_id)] = sentences

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
    for i, (chapter_id, frag_id) in enumerate(chapter_fragment_pairs):
        # Check if there are skipped fragments before this one (within same chapter)
        if i > 0:
            prev_chapter_id, prev_frag_id = chapter_fragment_pairs[i - 1]
            # Only insert summaries if in same chapter with gap
            if chapter_id == prev_chapter_id and frag_id > prev_frag_id + 1:
                skipped_summaries = []
                for skipped_id in range(prev_frag_id + 1, frag_id):
                    summary = fragment_reader.get_summary(chapter_id, skipped_id)
                    if summary:  # Only add non-empty summaries
                        skipped_summaries.append(summary)

                # Combine all skipped summaries into one natural paragraph
                if skipped_summaries:
                    result_parts.append(" ".join(skipped_summaries))

        # Render the current fragment
        marked_text = _build_fragment_markup(
            chapter_id,
            frag_id,
            fragment_sentences[(chapter_id, frag_id)],
            chunk_coverage,
            wrap_high_retention,
            full_markup,
        )
        result_parts.append(marked_text)

    return "\n\n".join(result_parts)


def _build_fragment_markup(
    chapter_id: int,
    frag_id: int,
    sentences: list[tuple[int, str]],
    chunk_coverage: dict[SentenceId, list],
    wrap_high_retention: bool = False,
    full_markup: bool = False,
) -> str:
    """Build marked-up text for one fragment.

    Args:
        chapter_id: Chapter ID
        frag_id: Fragment ID
        sentences: List of (sentence_index, text) tuples
        chunk_coverage: Map from sentence_id to list of Chunk objects
        wrap_high_retention: If True, wrap only high-retention chunks (verbatim/detailed) with simplified tags
        full_markup: If True, wrap ALL chunks with full XML attributes

    Returns:
        Fragment text with <chunk> markup
    """
    # Group consecutive sentences into ranges
    # Each range is either: unmarked text OR chunk-marked text
    sentence_ranges: list[tuple[int, int, dict | None]] = []  # (start_idx, end_idx, chunk_attrs or None)

    i = 0
    while i < len(sentences):
        sent_idx, sent_text = sentences[i]
        sid = (chapter_id, frag_id, sent_idx)

        if sid in chunk_coverage:
            # Start of a chunk range
            chunk_attrs = _merge_chunk_attributes(chunk_coverage[sid])
            start_idx = i

            # Find consecutive sentences in same chunk(s)
            j = i + 1
            while j < len(sentences):
                next_sid = (chapter_id, frag_id, sentences[j][0])
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
            retention = chunk_attrs.get("retention")
            importance = chunk_attrs.get("importance")
            label = chunk_attrs.get("label")

            if full_markup:
                # Full markup mode: wrap ALL chunks with complete XML attributes
                attrs_list = [f'label="{label}"']
                if retention:
                    attrs_list.append(f'retention="{retention}"')
                if importance:
                    attrs_list.append(f'importance="{importance}"')
                attrs_str = " ".join(attrs_list)
                parts.append(f"<chunk {attrs_str}>{text_segment}</chunk>")
            elif wrap_high_retention and retention in ("verbatim", "detailed"):
                # Simplified high-retention mode: only wrap verbatim/detailed chunks with retention attribute
                parts.append(f'<chunk retention="{retention}">{text_segment}</chunk>')
            else:
                # No wrapping: plain text
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
