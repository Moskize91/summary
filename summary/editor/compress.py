"""Text compression with multi-reviewer iterative refinement."""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from json_repair import repair_json

from ..llm import LLM, LLMessage
from ..topologization import Topologization
from .clue import Clue, extract_clues_from_topologization
from .markup import format_clue_as_book


def _extract_json_from_markdown(text: str) -> str:
    """Extract JSON from markdown code blocks.

    Args:
        text: Text that may contain markdown code blocks

    Returns:
        Extracted JSON string (or original text if no code block found)
    """
    # Try to extract JSON from markdown code block
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_compressed_text(full_response: str) -> str:
    """Extract the actual compressed text from AI's structured response.

    The AI may output in this format:
    ## Working Notes (internal)
    [thoughts...]
    ---
    ## Compressed Text (or 压缩文本)
    [actual compressed text]
    ---
    [additional notes...]

    Args:
        full_response: Full response from AI

    Returns:
        Extracted compressed text only
    """
    # Try to find "## Compressed Text" or "## 压缩文本" section
    # Support both English and Chinese headings
    match = re.search(
        r"##\s*(?:Compressed\s+Text|压缩文本)\s*\n+(.*?)(?:\n+---|\*\*CRITICAL\*\*|$)",
        full_response,
        re.DOTALL | re.IGNORECASE,
    )

    if match:
        compressed_text = match.group(1).strip()
        # Remove any 【】 brackets if present at start/end
        compressed_text = re.sub(r"^【|】$", "", compressed_text).strip()
        return compressed_text

    # Fallback: if no section marker found, return the full response
    # (for backward compatibility with old format)
    return full_response.strip()


def _extract_thinking_text(full_response: str) -> str:
    """Extract thinking/working notes from AI response, excluding the compressed text section.

    Args:
        full_response: Full response from AI

    Returns:
        Thinking text only (everything before the compressed text section)
    """
    # Find the compressed text section
    match = re.search(
        r"##\s*(?:Compressed\s+Text|压缩文本)\s*",
        full_response,
        re.IGNORECASE,
    )

    if match:
        # Get everything before the compressed text section
        thinking = full_response[: match.start()].strip()
        return thinking

    # If no section found, return empty (backward compatibility)
    return ""


def _clean_chunk_tags(text: str) -> str:
    """Remove all <chunk> XML tags from text.

    AI sometimes includes <chunk retention="..."> tags in the output even though
    they should only be used for input marking. This function removes all such tags
    while preserving the text content inside.

    Args:
        text: Text that may contain <chunk> tags

    Returns:
        Cleaned text with all <chunk> tags removed
    """
    # Remove opening tags: <chunk> or <chunk retention="...">
    text = re.sub(r'<chunk(?:\s+retention="[^"]*")?>', "", text)
    # Remove closing tags: </chunk>
    text = re.sub(r"</chunk>", "", text)
    return text


@dataclass
class ClueReviewerInfo:
    """Information about a clue for reviewing compressed text."""

    clue_id: int
    weight: float
    label: str  # Simple label like "first_label → last_label"
    reviewer_info: str  # Natural language review guidelines


@dataclass
class ReviewResult:
    """Result from a single reviewer."""

    clue_id: int
    weight: float
    issues: list[dict]  # List of issue dicts with type, severity, missing_info/problem, suggestion


@dataclass
class CompressionVersion:
    """A single compression version with its score."""

    iteration: int
    text: str
    score: float  # Lower is better
    reviews: list[ReviewResult]


def _format_chunk_hierarchy(topologization: Topologization, chapter_id: int, group_id: int) -> str:
    """Format chunk hierarchy for logging: Group -> Snakes -> Chunks.

    Shows the 3-level structure:
    - Group (chapter_id + group_id)
    - Snakes (with snake_id, weight, label)
    - Chunks (with retention, importance, content preview)

    Args:
        topologization: Topologization object
        chapter_id: Chapter ID
        group_id: Group ID

    Returns:
        Formatted hierarchy string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"CHUNK HIERARCHY - Chapter {chapter_id}, Group {group_id}")
    lines.append("=" * 80)
    lines.append("")

    # Get snake graph for this group
    snake_graph = topologization.get_snake_graph(chapter_id, group_id)
    snakes = list(snake_graph)

    if not snakes:
        lines.append("(No snakes found in this group)")
        lines.append("")
        return "\n".join(lines)

    # Sort snakes by weight descending
    snakes.sort(key=lambda s: s.weight, reverse=True)

    for snake_idx, snake in enumerate(snakes, 1):
        # Snake header
        lines.append(f"Snake #{snake_idx} (ID: {snake.snake_id})")
        lines.append(f"├─ Weight: {snake.weight:.4f}")
        lines.append(f"├─ Label: {snake.first_label} → {snake.last_label}")
        lines.append(f"├─ Chunks: {snake.size}")
        lines.append(f"└─ Tokens: {snake.tokens}")
        lines.append("")

        # Get chunks in this snake
        chunks = snake.get_chunks()

        for chunk_idx, chunk in enumerate(chunks, 1):
            is_last_chunk = chunk_idx == len(chunks)
            prefix = "    └─" if is_last_chunk else "    ├─"
            continuation = "      " if is_last_chunk else "    │ "

            # Chunk info
            lines.append(f"{prefix} Chunk {chunk_idx}/{len(chunks)} (ID: {chunk.id})")
            lines.append(f"{continuation} • Label: {chunk.label}")
            lines.append(f"{continuation} • Retention: {chunk.retention or 'N/A'}")
            lines.append(f"{continuation} • Importance: {chunk.importance or 'N/A'}")

            # Content preview (first 60 chars)
            content_preview = chunk.content[:60] + "..." if len(chunk.content) > 60 else chunk.content
            lines.append(f"{continuation} • Content: {content_preview}")

            if not is_last_chunk:
                lines.append(f"{continuation}")

        lines.append("")

    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


async def compress_text(
    topologization: Topologization,
    chapter_id: int,
    group_id: int,
    intention: str,
    llm: LLM,
    compression_ratio: float = 0.2,
    max_iterations: int = 5,
    max_clues: int = 10,
    log_dir_path: Path | None = None,
) -> str:
    """Compress text from a specific fragment group using iterative refinement.

    Args:
        topologization: Topologization object with knowledge graph and snakes
        chapter_id: Chapter ID to compress
        group_id: Group ID within the chapter to compress
        intention: User's reading intention
        llm: LLM instance
        compression_ratio: Target compression ratio (default: 0.2 = 20%)
        max_iterations: Number of iterations (default: 5)
        max_clues: Maximum number of clues (narrative threads) to generate (default: 10)
        log_dir_path: Directory for compression logs (default: None, no logging)

    Returns:
        Compressed text string
    """
    print("\n" + "=" * 60)
    print("=== Text Compression Pipeline ===")
    print(f"=== Chapter {chapter_id}, Group {group_id} ===")
    print("=" * 60)

    # Setup logging
    log_file = None
    if log_dir_path is not None:
        log_dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        log_file = log_dir_path / f"compression chapter-{chapter_id} group-{group_id} {timestamp}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=== Text Compression Log ===\n")
            f.write(f"Chapter: {chapter_id}, Group: {group_id}\n")
            f.write(f"Started at: {timestamp}\n")
            f.write(f"Compression ratio target: {compression_ratio:.0%}\n")
            f.write(f"Max iterations: {max_iterations}\n")
            f.write("\n\n")

            # Write chunk hierarchy
            hierarchy = _format_chunk_hierarchy(topologization, chapter_id, group_id)
            f.write(hierarchy)
            f.write("\n")

    # Step 1: Get original text for this group
    print("\nStep 1: Loading original text...")
    original_text = _get_full_text(topologization, chapter_id, group_id)
    print(f"Original text length: {len(original_text)} characters")

    # Step 2: Extract clues from this group's snakes
    print("\nStep 2: Extracting clues from topologization...")
    clues = extract_clues_from_topologization(
        topologization, max_clues=max_clues, chapter_id=chapter_id, group_id=group_id
    )

    # Count snakes in this group
    snake_graph = topologization.get_snake_graph(chapter_id, group_id)
    total_snakes = len(list(snake_graph))

    print(f"Extracted {len(clues)} clues (max: {max_clues}) from {total_snakes} snakes")

    # Step 2.5: Generate marked text for compression (with high-retention chunks wrapped)
    print("\nStep 2.5: Generating marked text for compression...")
    marked_original_text = _get_marked_full_text(topologization, clues, chapter_id, group_id)
    print(f"Marked text length: {len(marked_original_text)} characters")

    # Step 3: Generate reviewer info for each clue
    print("\nStep 3: Generating clue reviewers...")
    clue_reviewers = await _generate_clue_reviewers(clues, llm, topologization)
    print(f"Generated {len(clue_reviewers)} clue reviewers")

    # Step 4: Calculate target length
    target_length = int(len(original_text) * compression_ratio)
    print(f"Target length: {target_length} characters ({compression_ratio:.0%} compression)")

    # Step 5: Iterative compression - always run max_iterations times
    print(f"\nStep 4: Iterative compression ({max_iterations} iterations)...")
    versions: list[CompressionVersion] = []
    previous_compressed_text: str | None = None
    revision_feedback: str | None = None
    previous_reviews: list[ReviewResult] | None = None  # Store reviews for logging in next iteration
    reviewer_histories: dict[int, tuple[str, str]] = {}  # clue_id -> (prev_compressed_text, prev_response)

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # Log iteration header and revision feedback BEFORE compression
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"ITERATION {iteration}/{max_iterations}\n")
                f.write(f"{'=' * 80}\n\n")

                # Show revision feedback if present (from previous iteration)
                if revision_feedback:
                    f.write("Revision Feedback (Compressor's View):\n")
                    f.write(f"{'-' * 80}\n")
                    f.write(revision_feedback)
                    f.write(f"\n{'-' * 80}\n\n")

        # 4.1 Compress text
        print("Compressing text...")
        full_response, compressed_text = await _compress_iteration(
            marked_text=marked_original_text,
            target_length=target_length,
            compression_ratio=compression_ratio,
            clue_reviewers=clue_reviewers,
            previous_compressed_text=previous_compressed_text,
            revision_feedback=revision_feedback,
            llm=llm,
        )
        print(f"Compressed to {len(compressed_text)} characters")

        # Log compressed text result
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as f:
                # Log thinking text (excluding compressed text section)
                thinking_text = _extract_thinking_text(full_response)
                if thinking_text and thinking_text.strip():
                    f.write("Thinking:\n")
                    f.write(f"{'-' * 80}\n")
                    f.write(thinking_text)
                    f.write(f"\n{'-' * 80}\n\n")

                # Log extracted compressed text with character count
                f.write(f"Compressed Text ({len(compressed_text)} characters):\n")
                f.write(f"{'-' * 80}\n")
                f.write(compressed_text)
                f.write(f"\n{'-' * 80}\n\n\n")

        # 4.2 Review with all reviewers
        print("Reviewing compressed text...")
        reviews, raw_responses = await _review_compression(
            compressed_text=compressed_text,
            clue_reviewers=clue_reviewers,
            llm=llm,
            reviewer_histories=reviewer_histories if reviewer_histories else None,
        )

        # 4.3 Calculate score (lower is better)
        score = _calculate_score(reviews)
        print(f"Issue score: {score:.2f} (lower is better)")

        # Store this version
        versions.append(
            CompressionVersion(
                iteration=iteration,
                text=compressed_text,
                score=score,
                reviews=reviews,
            )
        )

        # If score is 0 (perfect), stop early
        if score == 0:
            print("✓ Perfect compression achieved (score = 0)")
            break

        # 4.5 Collect feedback for next iteration
        if iteration < max_iterations:
            print("Preparing revision feedback...")

            revision_feedback = _collect_feedback(reviews, llm)
            previous_compressed_text = compressed_text

            # Update reviewer histories for next iteration
            for clue_id, raw_response in raw_responses.items():
                reviewer_histories[clue_id] = (compressed_text, raw_response)

    # Step 5: Select best version (lowest score)
    if not versions:
        raise RuntimeError("Compression failed: no versions generated")

    best_version = min(versions, key=lambda v: v.score)
    print(f"\n✓ Selected iteration {best_version.iteration} with score {best_version.score:.2f}")

    # Log final selection
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write("FINAL SELECTION\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"Selected: Iteration {best_version.iteration}/{max_iterations}\n")
            f.write(f"Score: {best_version.score:.2f}\n")
            f.write(f"Length: {len(best_version.text)} characters\n")
            f.write(f"Compression ratio: {len(best_version.text) / len(original_text):.1%}\n")
            f.write(f"\n{'=' * 80}\n\n")

            # Print remaining unresolved issues
            if best_version.score > 0:
                f.write("REMAINING UNRESOLVED ISSUES\n")
                f.write(f"{'=' * 80}\n\n")

                issue_number = 1
                for review in best_version.reviews:
                    for issue in review.issues:
                        severity = issue.get("severity", "minor").upper()
                        problem = issue.get("problem", "")
                        suggestion = issue.get("suggestion", "")

                        f.write(f"{issue_number}. [{severity}]\n")
                        f.write(f"   Problem: {problem}\n")
                        if suggestion:
                            f.write(f"   Suggestion: {suggestion}\n")
                        f.write("\n")
                        issue_number += 1

                f.write(f"{'=' * 80}\n\n")
            else:
                f.write("🎉 CONGRATULATIONS! 🎉\n")
                f.write(f"{'=' * 80}\n\n")
                f.write("All issues have been successfully resolved!\n")
                f.write("The compressed text meets all quality requirements.\n")
                f.write(f"\n{'=' * 80}\n\n")

    print("\n" + "=" * 60)
    print("=== Compression Complete ===")
    print("=" * 60)
    print(f"Final length: {len(best_version.text)} characters")
    print(f"Compression ratio: {len(best_version.text) / len(original_text):.1%}")

    return best_version.text


def _get_full_text(topologization: Topologization, chapter_id: int, group_id: int) -> str:
    """Get full original text for a specific fragment group.

    Args:
        topologization: Topologization object
        chapter_id: Chapter ID
        group_id: Group ID within the chapter

    Returns:
        Full text string for this group's fragments
    """
    # Get fragment IDs for this group from database
    conn = topologization._conn
    cursor = conn.execute(
        """
        SELECT fragment_id FROM fragment_groups
        WHERE chapter_id = ? AND group_id = ?
        ORDER BY fragment_id
        """,
        (chapter_id, group_id),
    )
    fragment_ids = [row[0] for row in cursor.fetchall()]

    if not fragment_ids:
        return ""

    # Read all fragments in this group
    fragments_dir = topologization.workspace_path / "fragments"
    chapter_dir = fragments_dir / f"chapter-{chapter_id}"

    all_sentences = []
    for fragment_id in fragment_ids:
        fragment_file = chapter_dir / f"fragment_{fragment_id}.json"

        with open(fragment_file, encoding="utf-8") as f:
            fragment_data = json.load(f)

        # Handle both old format (list) and new format (dict with "sentences")
        if isinstance(fragment_data, list):
            sentences = fragment_data
        else:
            sentences = fragment_data["sentences"]

        for sentence in sentences:
            all_sentences.append(sentence["text"])

    return " ".join(all_sentences)


def _get_marked_full_text(topologization: Topologization, clues: list[Clue], chapter_id: int, group_id: int) -> str:
    """Get full original text with high-retention chunks marked.

    Wraps verbatim/detailed retention chunks in <chunk retention="XXX"> tags
    to signal the compressor that these parts need special preservation.

    Args:
        topologization: Topologization object
        clues: List of Clue objects with chunks
        chapter_id: Chapter ID (not used, kept for consistency)
        group_id: Group ID (not used, kept for consistency)

    Returns:
        Full text string with high-retention chunks wrapped in <chunk> tags
    """
    # Collect all chunks from all clues
    all_chunks = []
    for clue in clues:
        all_chunks.extend(clue.chunks)

    # Use format_clue_as_book with wrap_high_retention=True
    # This will format the entire text with only high-retention chunks marked
    marked_text = format_clue_as_book(all_chunks, topologization, wrap_high_retention=True)

    return marked_text


async def _generate_clue_reviewers(
    clues: list[Clue],
    llm: LLM,
    topologization: Topologization,
) -> list[ClueReviewerInfo]:
    """Generate reviewer info for each clue.

    Args:
        clues: List of Clue objects (already weight-normalized)
        llm: LLM instance
        topologization: Topologization object (for formatting chunks)

    Returns:
        List of ClueReviewerInfo objects
    """
    clue_reviewers = []

    # Find prompt template
    info_prompt_path = Path(__file__).parent.parent / "data" / "editor" / "clue_reviewer_generator.jinja"

    for clue in clues:
        print(f"  Processing clue {clue.clue_id}: {clue.label}")
        print(f"    Weight: {clue.weight:.2f}")
        if clue.is_merged:
            print(f"    Merged from snakes: {clue.source_snake_ids}")

        # Format chunks as book-like text with XML markup
        clue_text = format_clue_as_book(clue.chunks, topologization, full_markup=True)

        # Generate reviewer strategy
        info_system_prompt = llm.load_system_prompt(info_prompt_path)
        user_message = clue_text

        info_response = await llm.request(
            system_prompt=info_system_prompt,
            user_message=user_message,
            temperature=0.3,
        )

        if not info_response:
            raise RuntimeError(f"Clue {clue.clue_id} reviewer info generation failed: LLM returned empty response")

        reviewer_info = info_response.strip()
        print(f"    Reviewer strategy generated ({len(reviewer_info)} chars)")

        clue_reviewers.append(
            ClueReviewerInfo(
                clue_id=clue.clue_id,
                weight=clue.weight,
                label=clue.label,
                reviewer_info=reviewer_info,
            )
        )

    print("\n  Clue weights (already normalized, sum = 1.0):")
    for cr in clue_reviewers:
        print(f"    Clue {cr.clue_id}: {cr.weight:.3f}")

    return clue_reviewers


async def _compress_iteration(
    marked_text: str,
    target_length: int,
    compression_ratio: float,
    clue_reviewers: list[ClueReviewerInfo],
    previous_compressed_text: str | None,
    revision_feedback: str | None,
    llm: LLM,
) -> tuple[str, str]:
    """Perform one compression iteration.

    Args:
        marked_text: Marked original text with high-retention chunks wrapped in <chunk> tags
        target_length: Target length in characters
        compression_ratio: Compression ratio
        clue_reviewers: List of clue reviewer info
        previous_compressed_text: Previous iteration's compressed text (None for first iteration)
        revision_feedback: Feedback from previous iteration (None for first iteration)
        llm: LLM instance

    Returns:
        Tuple of (full_response, compressed_text):
        - full_response: Complete AI response including Working Notes
        - compressed_text: Extracted compressed text only
    """
    # Load prompt template (no longer includes revision_feedback)
    prompt_path = Path(__file__).parent.parent / "data" / "editor" / "text_compressor.jinja"

    # Calculate acceptable length range (±15% of target)
    acceptable_min = int(target_length * 0.85)
    acceptable_max = int(target_length * 1.15)

    system_prompt = llm.load_system_prompt(
        prompt_path,
        original_length=len(marked_text),
        target_length=target_length,
        compression_ratio=int(compression_ratio * 100),
        acceptable_min=acceptable_min,
        acceptable_max=acceptable_max,
    )

    # Build messages based on iteration
    # User message contains the marked original text
    user_message = marked_text

    if previous_compressed_text is None or revision_feedback is None:
        # First iteration: simple request
        response = await llm.request(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.5,
        )
    else:
        # Subsequent iterations: include previous attempt and feedback
        messages = [
            LLMessage(role="system", content=system_prompt),
            LLMessage(role="user", content=user_message),
            LLMessage(role="assistant", content=previous_compressed_text),
            LLMessage(role="user", content=revision_feedback),
        ]
        response = await llm.request_with_history(
            messages=messages,
            temperature=0.5,
        )

    if not response:
        raise RuntimeError("Compression failed: LLM returned empty response")

    full_response = response.strip()

    # Extract compressed text section from full response
    compressed_text = _extract_compressed_text(full_response)

    # Clean up any <chunk> tags that AI might have included
    compressed_text = _clean_chunk_tags(compressed_text)

    return full_response, compressed_text


async def _review_compression(
    compressed_text: str,
    clue_reviewers: list[ClueReviewerInfo],
    llm: LLM,
    reviewer_histories: dict[int, tuple[str, str]] | None = None,
) -> tuple[list[ReviewResult], dict[int, str]]:
    """Review compressed text with all clue reviewers.

    Args:
        compressed_text: Compressed text to review
        clue_reviewers: List of clue reviewer info
        llm: LLM instance
        reviewer_histories: Optional dict mapping clue_id to (previous_compressed_text, previous_response)

    Returns:
        Tuple of (reviews, raw_responses):
        - reviews: List of ReviewResult objects
        - raw_responses: Dict mapping clue_id to raw response text
    """
    reviews = []
    raw_responses = {}
    prompt_path = Path(__file__).parent.parent / "data" / "editor" / "clue_reviewer.jinja"

    for cr in clue_reviewers:
        system_prompt = llm.load_system_prompt(
            prompt_path,
            thread_info=cr.reviewer_info,
        )

        # Check if this reviewer has history
        if reviewer_histories is not None and cr.clue_id in reviewer_histories:
            prev_compressed_text, prev_response = reviewer_histories[cr.clue_id]
            # Build conversation history: system + user(prev) + assistant(prev) + user(current)
            # All user messages contain only compressed text, no intention
            messages = [
                LLMessage(role="system", content=system_prompt),
                LLMessage(role="user", content=prev_compressed_text),
                LLMessage(role="assistant", content=prev_response),
                LLMessage(role="user", content=compressed_text),
            ]
            response = await llm.request_with_history(
                messages=messages,
                temperature=0.3,
            )
        else:
            # First iteration: simple request with only compressed text
            response = await llm.request(
                system_prompt=system_prompt,
                user_message=compressed_text,
                temperature=0.3,
            )

        if not response:
            raise RuntimeError(f"Clue {cr.clue_id} review failed: LLM returned empty response")

        # Store raw response
        raw_responses[cr.clue_id] = response

        try:
            review_json = _extract_json_from_markdown(response)
            review_data = json.loads(repair_json(review_json))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Clue {cr.clue_id} review failed: JSON parse error - {e}") from e

        try:
            issues = review_data.get("issues", [])
            reviews.append(
                ReviewResult(
                    clue_id=cr.clue_id,
                    weight=cr.weight,
                    issues=issues,
                )
            )
            print(f"  Clue {cr.clue_id}: {len(issues)} issues")
        except (ValueError, KeyError) as e:
            raise RuntimeError(f"Clue {cr.clue_id} review failed: Invalid data format - {e}") from e

    return reviews, raw_responses


def _calculate_score(reviews: list[ReviewResult]) -> float:
    """Calculate issue score from reviews (lower is better).

    Score = sum of (severity * weight) for all issues
    - critical: 9
    - major: 3
    - minor: 1

    Args:
        reviews: List of ReviewResult objects

    Returns:
        Issue score (0 = perfect, higher = more problems)
    """
    severity_values = {"critical": 9, "major": 3, "minor": 1}

    total_score = 0.0
    for review in reviews:
        for issue in review.issues:
            severity = issue.get("severity", "minor").lower()
            severity_value = severity_values.get(severity, 1)
            total_score += severity_value * review.weight

    return total_score


def _format_issues_for_log(reviews: list[ReviewResult]) -> str:
    """Format all issues for logging (including hidden ones), with visual separation.

    Args:
        reviews: List of ReviewResult objects

    Returns:
        Formatted string with all issues, visually separated between visible and hidden
    """
    severity_values = {"critical": 9, "major": 3, "minor": 1}

    # Collect all issues with metadata
    all_issues = []
    for review in reviews:
        for issue_index, issue in enumerate(review.issues):
            severity = issue.get("severity", "minor").lower()
            severity_value = severity_values.get(severity, 1)
            all_issues.append(
                {
                    "clue_id": review.clue_id,
                    "issue_index": issue_index,
                    "severity": severity,
                    "severity_value": severity_value,
                    "weight": review.weight,
                    "description": issue.get("missing_info") or issue.get("problem", "No description"),
                    "suggestion": issue.get("suggestion", ""),
                }
            )

    if not all_issues:
        return "No issues found - all reviewers are satisfied."

    # Sort by (severity_value DESC, weight DESC, issue_index ASC)
    all_issues.sort(key=lambda x: (-x["severity_value"], -x["weight"], x["issue_index"]))

    # Format all issues with visual separation
    issues_lines = []
    visible_count = min(9, len(all_issues))

    # Format visible issues (AI can see these)
    for i, issue in enumerate(all_issues[:visible_count], 1):
        issues_lines.append(f"{i}. [{issue['severity'].upper()}]")
        issues_lines.append(f"   Problem: {issue['description']}")
        if issue["suggestion"]:
            issues_lines.append(f"   Suggestion: {issue['suggestion']}")
        issues_lines.append("")

    # Add separator and hidden issues if any
    if len(all_issues) > visible_count:
        issues_lines.append("=" * 80)
        issues_lines.append("HIDDEN ISSUES (AI cannot see below, for debugging only)")
        issues_lines.append("=" * 80)
        issues_lines.append("")

        for i, issue in enumerate(all_issues[visible_count:], visible_count + 1):
            issues_lines.append(f"{i}. [{issue['severity'].upper()}]")
            issues_lines.append(f"   Problem: {issue['description']}")
            if issue["suggestion"]:
                issues_lines.append(f"   Suggestion: {issue['suggestion']}")
            issues_lines.append("")

    return "\n".join(issues_lines)


def _collect_feedback(reviews: list[ReviewResult], llm: LLM) -> str:
    """Collect feedback from reviews for next iteration.

    Implements blind review: issues are aggregated and sorted by severity/importance,
    without revealing which snake raised them.

    Args:
        reviews: List of ReviewResult objects
        llm: LLM instance for loading template

    Returns:
        Formatted feedback string for compressor
    """
    severity_values = {"critical": 9, "major": 3, "minor": 1}

    # Collect all issues with metadata
    all_issues = []
    for review in reviews:
        for issue_index, issue in enumerate(review.issues):
            severity = issue.get("severity", "minor").lower()
            severity_value = severity_values.get(severity, 1)
            all_issues.append(
                {
                    "clue_id": review.clue_id,
                    "issue_index": issue_index,
                    "severity": severity,
                    "severity_value": severity_value,
                    "weight": review.weight,
                    "description": issue.get("missing_info") or issue.get("problem", "No description"),
                    "suggestion": issue.get("suggestion", ""),
                }
            )

    if not all_issues:
        return "No issues found - all reviewers are satisfied."

    # Sort by (severity_value DESC, weight DESC, issue_index ASC)
    all_issues.sort(key=lambda x: (-x["severity_value"], -x["weight"], x["issue_index"]))

    # Format issues description (max 9 issues shown) - for LLM feedback
    issues_lines = []
    visible_count = min(9, len(all_issues))
    hidden_count = len(all_issues) - visible_count

    for i, issue in enumerate(all_issues[:visible_count], 1):
        issues_lines.append(f"{i}. [{issue['severity'].upper()}]")
        issues_lines.append(f"   Problem: {issue['description']}")
        if issue["suggestion"]:
            issues_lines.append(f"   Suggestion: {issue['suggestion']}")
        issues_lines.append("")

    if hidden_count > 0:
        issues_lines.append(f"... and {hidden_count} more issues hidden (lower priority)")

    issues_description = "\n".join(issues_lines)

    # Load feedback template and render
    feedback_template_path = Path(__file__).parent.parent / "data" / "editor" / "revision_feedback.jinja"
    feedback_message = llm.load_system_prompt(
        feedback_template_path,
        issues_description=issues_description,
    )

    return feedback_message
