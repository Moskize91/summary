"""Text compression with multi-reviewer iterative refinement."""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from json_repair import repair_json

from ..llm import LLM
from ..topologization.api import Topologization


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


@dataclass
class SnakeReviewerInfo:
    """Information about a snake for reviewing compressed text."""

    snake_id: int
    weight: float
    label: str  # Simple label like "first_label → last_label"
    reviewer_info: str  # Natural language review guidelines


@dataclass
class ReviewResult:
    """Result from a single reviewer."""

    snake_id: int
    weight: float
    issues: list[dict]  # List of issue dicts with type, severity, missing_info/problem, suggestion


@dataclass
class CompressionVersion:
    """A single compression version with its score."""

    iteration: int
    text: str
    score: float  # Lower is better
    reviews: list[ReviewResult]


def compress_text(
    topologization: Topologization,
    intention: str,
    llm: LLM,
    compression_ratio: float = 0.2,
    max_iterations: int = 5,
    log_dir_path: Path | None = None,
) -> str:
    """Compress text from topologization using iterative refinement.

    Args:
        topologization: Topologization object with knowledge graph and snakes
        intention: User's reading intention
        llm: LLM instance
        compression_ratio: Target compression ratio (default: 0.2 = 20%)
        max_iterations: Number of iterations (default: 5)
        log_dir_path: Directory for compression logs (default: None, no logging)

    Returns:
        Compressed text string
    """
    print("\n" + "=" * 60)
    print("=== Text Compression Pipeline ===")
    print("=" * 60)

    # Setup logging
    log_file = None
    if log_dir_path is not None:
        log_dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        log_file = log_dir_path / f"compression {timestamp}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=== Text Compression Log ===\n")
            f.write(f"Started at: {timestamp}\n")
            f.write(f"Compression ratio target: {compression_ratio:.0%}\n")
            f.write(f"Max iterations: {max_iterations}\n")
            f.write("\n\n")

    # Step 1: Get original text
    print("\nStep 1: Loading original text...")
    original_text = _get_full_text(topologization)
    print(f"Original text length: {len(original_text)} characters")

    # Step 2: Generate reviewer info for each snake
    print("\nStep 2: Generating snake reviewers...")
    snake_reviewers = _generate_snake_reviewers(topologization, intention, llm)
    print(f"Generated {len(snake_reviewers)} snake reviewers")

    # Step 3: Calculate target length
    target_length = int(len(original_text) * compression_ratio)
    print(f"Target length: {target_length} characters ({compression_ratio:.0%} compression)")

    # Step 4: Iterative compression - always run max_iterations times
    print(f"\nStep 3: Iterative compression ({max_iterations} iterations)...")
    versions: list[CompressionVersion] = []
    previous_compressed_text: str | None = None
    revision_feedback: str | None = None
    reviewer_histories: dict[int, tuple[str, str]] = {}  # snake_id -> (prev_compressed_text, prev_response)

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # 4.1 Compress text
        print("Compressing text...")
        full_response, compressed_text = _compress_iteration(
            original_text=original_text,
            target_length=target_length,
            compression_ratio=compression_ratio,
            snake_reviewers=snake_reviewers,
            previous_compressed_text=previous_compressed_text,
            revision_feedback=revision_feedback,
            intention=intention,
            llm=llm,
        )
        print(f"Compressed to {len(compressed_text)} characters")

        # Log compressed text
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"ITERATION {iteration}/{max_iterations}\n")
                f.write(f"{'=' * 80}\n\n")

                # Show revision feedback if present
                if revision_feedback:
                    f.write("Revision Feedback (Compressor's View):\n")
                    f.write(f"{'-' * 80}\n")
                    f.write(revision_feedback)
                    f.write(f"\n{'-' * 80}\n\n")

                # Log thinking text (excluding compressed text section)
                thinking_text = _extract_thinking_text(full_response)
                if thinking_text and thinking_text.strip():
                    f.write("Thinking:\n")
                    f.write(f"{'-' * 80}\n")
                    f.write(thinking_text)
                    f.write(f"\n{'-' * 80}\n\n")

                # Log extracted compressed text with character count
                f.write(f"Extracted Compressed Text ({len(compressed_text)} characters):\n")
                f.write(f"{'-' * 80}\n")
                f.write(compressed_text)
                f.write(f"\n{'-' * 80}\n\n\n")

        # 4.2 Review with all reviewers
        print("Reviewing compressed text...")
        reviews, raw_responses = _review_compression(
            compressed_text=compressed_text,
            snake_reviewers=snake_reviewers,
            intention=intention,
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

        # Log review results
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("Review Results:\n")
                f.write(f"{'-' * 80}\n")
                f.write(f"Issue Score: {score:.2f} (lower is better)\n\n")

                # Log all snake reviewers (including failed ones)
                for sr in snake_reviewers:
                    # Find corresponding review result
                    review = next((r for r in reviews if r.snake_id == sr.snake_id), None)

                    f.write(f"Snake {sr.snake_id} (weight: {sr.weight:.2f}):\n")
                    f.write(f"  Label: {sr.label}\n")
                    f.write(f"  Reviewer:\n{sr.reviewer_info}\n")

                    if review is None:
                        f.write("  ❌ REVIEW FAILED - No response from LLM or parse error\n")
                    else:
                        if review.issues:
                            f.write(f"\n[[ Issues ({len(review.issues)}) ]]\n")
                            for issue in review.issues:
                                tier = issue.get("tier", "?")
                                issue_type = issue.get("type", "unknown")
                                severity = issue.get("severity", "unknown")
                                description = issue.get("missing_info") or issue.get("problem", "No description")
                                suggestion = issue.get("suggestion", "")

                                f.write(f"    - [TIER {tier}] [{severity.upper()}] ({issue_type})\n")
                                f.write(f"      Problem: {description}\n")
                                if suggestion:
                                    f.write(f"      Suggestion: {suggestion}\n")
                        else:
                            f.write("  No issues reported\n")

                    f.write("\n")

                # Add decision summary
                f.write(f"{'-' * 80}\n")
                f.write("Decision: ")
                if score == 0:
                    f.write("✓ PERFECT - No issues found, compression successful\n")
                elif iteration < max_iterations:
                    f.write(
                        f"⟳ CONTINUE - Score {score:.2f}, proceeding to iteration {iteration + 1}/{max_iterations}\n"
                    )
                else:
                    f.write("⏹ FINAL - This is the last iteration\n")
                f.write(f"{'-' * 80}\n\n\n")

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
            for snake_id, raw_response in raw_responses.items():
                reviewer_histories[snake_id] = (compressed_text, raw_response)

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

    print("\n" + "=" * 60)
    print("=== Compression Complete ===")
    print("=" * 60)
    print(f"Final length: {len(best_version.text)} characters")
    print(f"Compression ratio: {len(best_version.text) / len(original_text):.1%}")

    return best_version.text


def _get_full_text(topologization: Topologization) -> str:
    """Get full original text by reading all fragments in order.

    Args:
        topologization: Topologization object

    Returns:
        Full text string
    """
    fragments_dir = topologization.workspace_path / "fragments"
    fragment_files = sorted(fragments_dir.glob("fragment_*.json"), key=lambda p: int(p.stem.split("_")[1]))

    all_sentences = []
    for fragment_file in fragment_files:
        import json

        with open(fragment_file, encoding="utf-8") as f:
            sentences = json.load(f)
            for sentence in sentences:
                all_sentences.append(sentence["text"])

    return " ".join(all_sentences)


def _generate_snake_reviewers(
    topologization: Topologization,
    intention: str,
    llm: LLM,
) -> list[SnakeReviewerInfo]:
    """Generate reviewer info for each snake.

    Args:
        topologization: Topologization object
        intention: User's reading intention
        llm: LLM instance

    Returns:
        List of SnakeReviewerInfo objects
    """
    snake_reviewers = []

    # Find prompt template
    info_prompt_path = Path(__file__).parent.parent / "data" / "editor" / "thread_reviewer_generator.jinja"

    for snake in topologization.snake_graph:
        print(f"  Processing snake {snake.snake_id}: {snake.first_label} → {snake.last_label}")

        # Read weight from snake data structure
        weight = snake.weight
        print(f"    Weight: {weight:.2f}")

        # Get chunks for this snake
        chunks = snake.get_chunks()

        # Format chunks as JSON with complete metadata
        chunks_json = _format_chunks_as_json(chunks, topologization)

        # Generate reviewer strategy
        info_system_prompt = llm.load_system_prompt(info_prompt_path)
        user_message = f"User's reading intention:\n{intention}\n\n---\n\nThread chunks:\n{chunks_json}"

        info_response = llm.request(
            system_prompt=info_system_prompt,
            user_message=user_message,
            temperature=0.3,
        )

        if not info_response:
            raise RuntimeError(f"Snake {snake.snake_id} reviewer info generation failed: LLM returned empty response")

        reviewer_info = info_response.strip()
        print(f"    Reviewer strategy generated ({len(reviewer_info)} chars)")

        # Generate simple label
        label = f"{snake.first_label} → {snake.last_label}"

        snake_reviewers.append(
            SnakeReviewerInfo(
                snake_id=snake.snake_id,
                weight=weight,
                label=label,
                reviewer_info=reviewer_info,
            )
        )

    # Normalize weights so they sum to 1.0
    total_weight = sum(sr.weight for sr in snake_reviewers)
    if total_weight > 0:
        for sr in snake_reviewers:
            sr.weight = sr.weight / total_weight
        print("\n  Normalized weights (sum = 1.0):")
        for sr in snake_reviewers:
            print(f"    Snake {sr.snake_id}: {sr.weight:.3f}")

    return snake_reviewers


def _format_chunks_as_json(chunks: list, topologization: Topologization) -> str:
    """Format chunks as JSON with complete metadata.

    Args:
        chunks: List of Chunk objects
        topologization: Topologization object

    Returns:
        JSON string with chunks metadata including retention, importance, and source sentences
    """
    chunks_with_metadata = []
    for chunk in chunks:
        # Get source sentences
        source_sentences = [topologization.get_sentence_text(sid) for sid in chunk.sentence_ids]

        chunks_with_metadata.append(
            {
                "chunk_id": chunk.id,
                "label": chunk.label,
                "content": chunk.content,
                "retention": chunk.retention,
                "importance": chunk.importance,
                "source_sentences": source_sentences,
            }
        )

    return json.dumps(chunks_with_metadata, ensure_ascii=False, indent=2)


def _compress_iteration(
    original_text: str,
    target_length: int,
    compression_ratio: float,
    snake_reviewers: list[SnakeReviewerInfo],
    previous_compressed_text: str | None,
    revision_feedback: str | None,
    intention: str,
    llm: LLM,
) -> tuple[str, str]:
    """Perform one compression iteration.

    Args:
        original_text: Original text to compress
        target_length: Target length in characters
        compression_ratio: Compression ratio
        snake_reviewers: List of snake reviewer info
        previous_compressed_text: Previous iteration's compressed text (None for first iteration)
        revision_feedback: Feedback from previous iteration (None for first iteration)
        intention: User's reading intention
        llm: LLM instance

    Returns:
        Tuple of (full_response, compressed_text):
        - full_response: Complete AI response including Working Notes
        - compressed_text: Extracted compressed text only
    """
    # Format thread summaries
    thread_summaries = "\n\n".join(
        [f"**Thread {sr.snake_id} (weight: {sr.weight:.2f}):**\n{sr.reviewer_info}" for sr in snake_reviewers]
    )

    # Load prompt template (no longer includes revision_feedback)
    prompt_path = Path(__file__).parent.parent / "data" / "editor" / "text_compressor.jinja"
    system_prompt = llm.load_system_prompt(
        prompt_path,
        original_length=len(original_text),
        target_length=target_length,
        compression_ratio=int(compression_ratio * 100),
        thread_summaries=thread_summaries,
    )

    # Build messages based on iteration
    # User message contains only the original text, no intention
    user_message = original_text

    if previous_compressed_text is None or revision_feedback is None:
        # First iteration: simple request
        response = llm.request(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.5,
        )
    else:
        # Subsequent iterations: include previous attempt and feedback
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": previous_compressed_text},
            {"role": "user", "content": revision_feedback},
        ]
        response = llm.request_with_history(
            messages=messages,
            temperature=0.5,
        )

    if not response:
        raise RuntimeError("Compression failed: LLM returned empty response")

    full_response = response.strip()
    compressed_text = _extract_compressed_text(full_response)

    return full_response, compressed_text


def _review_compression(
    compressed_text: str,
    snake_reviewers: list[SnakeReviewerInfo],
    intention: str,
    llm: LLM,
    reviewer_histories: dict[int, tuple[str, str]] | None = None,
) -> tuple[list[ReviewResult], dict[int, str]]:
    """Review compressed text with all snake reviewers.

    Args:
        compressed_text: Compressed text to review
        snake_reviewers: List of snake reviewer info
        intention: User's reading intention
        llm: LLM instance
        reviewer_histories: Optional dict mapping snake_id to (previous_compressed_text, previous_response)

    Returns:
        Tuple of (reviews, raw_responses):
        - reviews: List of ReviewResult objects
        - raw_responses: Dict mapping snake_id to raw response text
    """
    reviews = []
    raw_responses = {}
    prompt_path = Path(__file__).parent.parent / "data" / "editor" / "thread_reviewer.jinja"

    for sr in snake_reviewers:
        system_prompt = llm.load_system_prompt(
            prompt_path,
            thread_info=sr.reviewer_info,
        )

        # Check if this reviewer has history
        has_history = reviewer_histories is not None and sr.snake_id in reviewer_histories

        if has_history:
            prev_compressed_text, prev_response = reviewer_histories[sr.snake_id]
            # Build conversation history: system + user(prev) + assistant(prev) + user(current)
            # All user messages contain only compressed text, no intention
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prev_compressed_text},
                {"role": "assistant", "content": prev_response},
                {"role": "user", "content": compressed_text},
            ]
            response = llm.request_with_history(
                messages=messages,
                temperature=0.3,
            )
        else:
            # First iteration: simple request with only compressed text
            response = llm.request(
                system_prompt=system_prompt,
                user_message=compressed_text,
                temperature=0.3,
            )

        if not response:
            raise RuntimeError(f"Snake {sr.snake_id} review failed: LLM returned empty response")

        # Store raw response
        raw_responses[sr.snake_id] = response

        try:
            review_json = _extract_json_from_markdown(response)
            review_data = json.loads(repair_json(review_json))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Snake {sr.snake_id} review failed: JSON parse error - {e}") from e

        try:
            issues = review_data.get("issues", [])
            reviews.append(
                ReviewResult(
                    snake_id=sr.snake_id,
                    weight=sr.weight,
                    issues=issues,
                )
            )
            print(f"  Snake {sr.snake_id}: {len(issues)} issues")
        except (ValueError, KeyError) as e:
            raise RuntimeError(f"Snake {sr.snake_id} review failed: Invalid data format - {e}") from e

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
        for raw_index, issue in enumerate(review.issues):
            severity = issue.get("severity", "minor").lower()
            severity_value = severity_values.get(severity, 1)
            all_issues.append(
                {
                    "severity": severity,
                    "severity_value": severity_value,
                    "weight": review.weight,
                    "raw_index": raw_index,
                    "type": issue.get("type", "unknown"),
                    "description": issue.get("missing_info") or issue.get("problem", "No description"),
                    "suggestion": issue.get("suggestion", ""),
                }
            )

    if not all_issues:
        return "No issues found - all reviewers are satisfied."

    # Sort by (severity_value DESC, weight DESC, raw_index ASC)
    all_issues.sort(key=lambda x: (-x["severity_value"], -x["weight"], x["raw_index"]))

    # Format issues description (max 9 issues shown)
    issues_lines = []
    visible_count = min(9, len(all_issues))
    hidden_count = len(all_issues) - visible_count

    for i, issue in enumerate(all_issues[:visible_count], 1):
        issues_lines.append(f"{i}. [{issue['severity'].upper()}] ({issue['type']})")
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
