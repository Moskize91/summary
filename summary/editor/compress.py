"""Text compression with multi-reviewer iterative refinement."""

import json
from dataclasses import dataclass
from pathlib import Path

from ..llm import LLM
from ..topologization.api import ChunkType, Topologization


@dataclass
class SnakeReviewerInfo:
    """Information about a snake for reviewing compressed text."""

    snake_id: int
    weight: float
    reviewer_info: str  # Natural language review guidelines


@dataclass
class ReviewResult:
    """Result from a single reviewer."""

    snake_id: int
    weight: float
    user_intent_score: float
    narrative_flow_score: float
    issues: list[dict]


def compress_text(
    topologization: Topologization,
    intention: str,
    llm: LLM,
    compression_ratio: float = 0.2,
    quality_threshold: float = 7.0,
    max_iterations: int = 3,
) -> str:
    """Compress text from topologization using iterative refinement.

    Args:
        topologization: Topologization object with knowledge graph and snakes
        intention: User's reading intention
        llm: LLM instance
        compression_ratio: Target compression ratio (default: 0.2 = 20%)
        quality_threshold: Minimum quality score to accept (default: 7.0)
        max_iterations: Maximum refinement iterations (default: 3)

    Returns:
        Compressed text string
    """
    print("\n" + "=" * 60)
    print("=== Text Compression Pipeline ===")
    print("=" * 60)

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

    # Step 4: Iterative compression
    print("\nStep 3: Iterative compression...")
    compressed_text: str = ""
    revision_feedback: str | None = None

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # 4.1 Compress text
        print("Compressing text...")
        compressed_text = _compress_iteration(
            original_text=original_text,
            target_length=target_length,
            compression_ratio=compression_ratio,
            snake_reviewers=snake_reviewers,
            revision_feedback=revision_feedback,
            intention=intention,
            llm=llm,
        )
        print(f"Compressed to {len(compressed_text)} characters")

        # 4.2 Review with all reviewers
        print("Reviewing compressed text...")
        reviews = _review_compression(
            compressed_text=compressed_text,
            snake_reviewers=snake_reviewers,
            intention=intention,
            llm=llm,
        )

        # 4.3 Calculate quality
        quality = _calculate_quality(reviews)
        print(f"Quality score: {quality:.2f}/10")

        # 4.4 Check if quality is acceptable
        if quality >= quality_threshold:
            print(f"✓ Quality threshold reached ({quality:.2f} >= {quality_threshold})")
            break

        # 4.5 Collect feedback for next iteration
        if iteration < max_iterations:
            print("Quality below threshold, preparing revision feedback...")
            revision_feedback = _collect_feedback(reviews)
        else:
            print("Max iterations reached. Using current version.")

    if not compressed_text:
        raise RuntimeError("Compression failed: no compressed text generated")

    print("\n" + "=" * 60)
    print("=== Compression Complete ===")
    print("=" * 60)
    print(f"Final length: {len(compressed_text)} characters")
    print(f"Compression ratio: {len(compressed_text) / len(original_text):.1%}")

    return compressed_text


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

    # Find prompt templates
    weight_prompt_path = Path(__file__).parent.parent / "data" / "editor" / "thread_weight_evaluator.jinja"
    info_prompt_path = Path(__file__).parent.parent / "data" / "editor" / "thread_reviewer_generator.jinja"

    for snake in topologization.snake_graph:
        print(f"  Processing snake {snake.snake_id}: {snake.first_label} → {snake.last_label}")

        # Get chunks for this snake
        chunks = snake.get_chunks()

        # Format chunks for prompts
        chunks_text = _format_chunks_for_prompt(chunks, topologization)

        # Generate weight
        weight_system_prompt = llm.load_system_prompt(weight_prompt_path)
        weight_response = llm.request(
            system_prompt=weight_system_prompt,
            user_message=f"{intention}\n\n---\n\n{chunks_text}",
            temperature=0.3,
        )

        if not weight_response:
            print("    Warning: Failed to generate weight, using default 0.5")
            weight = 0.5
        else:
            try:
                weight_data = json.loads(weight_response)
                weight = float(weight_data.get("weight", 0.5))
                print(f"    Weight: {weight:.2f}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"    Warning: Failed to parse weight ({e}), using default 0.5")
                weight = 0.5

        # Generate reviewer info
        info_system_prompt = llm.load_system_prompt(info_prompt_path)
        info_response = llm.request(
            system_prompt=info_system_prompt,
            user_message=f"{intention}\n\n---\n\n{chunks_text}",
            temperature=0.3,
        )

        if not info_response:
            print("    Warning: Failed to generate reviewer info")
            reviewer_info = f"Thread {snake.snake_id}: {snake.first_label} → {snake.last_label}"
        else:
            reviewer_info = info_response.strip()
            print(f"    Reviewer info generated ({len(reviewer_info)} chars)")

        snake_reviewers.append(
            SnakeReviewerInfo(
                snake_id=snake.snake_id,
                weight=weight,
                reviewer_info=reviewer_info,
            )
        )

    return snake_reviewers


def _format_chunks_for_prompt(chunks: list, topologization: Topologization) -> str:
    """Format chunks for LLM prompts.

    Args:
        chunks: List of Chunk objects
        topologization: Topologization object

    Returns:
        Formatted string with chunks and source sentences
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        chunk_type = "user_focused" if chunk.type == ChunkType.USER_FOCUSED else "book_coherence"
        lines.append(f"## Chunk {i}")
        lines.append(f"**Label:** {chunk.label}")
        lines.append(f"**Type:** {chunk_type}")
        lines.append(f"**Content:** {chunk.content}")
        lines.append("")

        # Add source sentences
        source_sentences = []
        for sentence_id in chunk.sentence_ids:
            sentence_text = topologization.get_sentence_text(sentence_id)
            source_sentences.append(sentence_text)

        lines.append(f"**Source sentences ({len(source_sentences)}):**")
        for j, sentence in enumerate(source_sentences, 1):
            lines.append(f"{j}. {sentence}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _compress_iteration(
    original_text: str,
    target_length: int,
    compression_ratio: float,
    snake_reviewers: list[SnakeReviewerInfo],
    revision_feedback: str | None,
    intention: str,
    llm: LLM,
) -> str:
    """Perform one compression iteration.

    Args:
        original_text: Original text to compress
        target_length: Target length in characters
        compression_ratio: Compression ratio
        snake_reviewers: List of snake reviewer info
        revision_feedback: Feedback from previous iteration (if any)
        intention: User's reading intention
        llm: LLM instance

    Returns:
        Compressed text string
    """
    # Format thread summaries
    thread_summaries = "\n\n".join(
        [f"**Thread {sr.snake_id} (weight: {sr.weight:.2f}):**\n{sr.reviewer_info}" for sr in snake_reviewers]
    )

    # Load prompt template
    prompt_path = Path(__file__).parent.parent / "data" / "editor" / "text_compressor.jinja"
    system_prompt = llm.load_system_prompt(
        prompt_path,
        original_length=len(original_text),
        target_length=target_length,
        compression_ratio=int(compression_ratio * 100),
        thread_summaries=thread_summaries,
        revision_feedback=revision_feedback,
    )

    # Call LLM
    user_message = f"{intention}\n\n---\n\n{original_text}"
    response = llm.request(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=0.5,
    )

    if not response:
        raise RuntimeError("Compression failed: LLM returned empty response")

    return response.strip()


def _review_compression(
    compressed_text: str,
    snake_reviewers: list[SnakeReviewerInfo],
    intention: str,
    llm: LLM,
) -> list[ReviewResult]:
    """Review compressed text with all snake reviewers.

    Args:
        compressed_text: Compressed text to review
        snake_reviewers: List of snake reviewer info
        intention: User's reading intention
        llm: LLM instance

    Returns:
        List of ReviewResult objects
    """
    reviews = []
    prompt_path = Path(__file__).parent.parent / "data" / "editor" / "thread_reviewer.jinja"

    for sr in snake_reviewers:
        system_prompt = llm.load_system_prompt(
            prompt_path,
            thread_weight=sr.weight,
            thread_info=sr.reviewer_info,
        )

        user_message = f"{intention}\n\n---\n\n{compressed_text}"
        response = llm.request(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.3,
        )

        if not response:
            print(f"  Warning: Snake {sr.snake_id} review failed")
            continue

        try:
            review_data = json.loads(response)
            reviews.append(
                ReviewResult(
                    snake_id=sr.snake_id,
                    weight=sr.weight,
                    user_intent_score=float(review_data.get("user_intent_score", 5.0)),
                    narrative_flow_score=float(review_data.get("narrative_flow_score", 5.0)),
                    issues=review_data.get("issues", []),
                )
            )
            print(
                f"  Snake {sr.snake_id}: "
                f"intent={review_data.get('user_intent_score', 5.0):.1f}, "
                f"flow={review_data.get('narrative_flow_score', 5.0):.1f}"
            )
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Warning: Snake {sr.snake_id} review parse failed: {e}")

    return reviews


def _calculate_quality(reviews: list[ReviewResult]) -> float:
    """Calculate weighted quality score from reviews.

    Args:
        reviews: List of ReviewResult objects

    Returns:
        Quality score (0-10)
    """
    if not reviews:
        return 0.0

    # Calculate weighted average
    # Combined score: 70% user intent, 30% narrative flow
    total_score = 0.0
    total_weight = 0.0

    for review in reviews:
        combined_score = review.user_intent_score * 0.7 + review.narrative_flow_score * 0.3
        total_score += combined_score * review.weight
        total_weight += review.weight

    if total_weight == 0:
        return 0.0

    return total_score / total_weight


def _collect_feedback(reviews: list[ReviewResult]) -> str:
    """Collect feedback from reviews for next iteration.

    Args:
        reviews: List of ReviewResult objects

    Returns:
        Formatted feedback string
    """
    feedback_lines = []

    for review in reviews:
        if not review.issues:
            continue

        feedback_lines.append(f"**Thread {review.snake_id} (weight: {review.weight:.2f}):**")
        feedback_lines.append(
            f"Scores: User Intent={review.user_intent_score:.1f}/10, "
            f"Narrative Flow={review.narrative_flow_score:.1f}/10"
        )
        feedback_lines.append("")

        for issue in review.issues:
            issue_type = issue.get("type", "unknown")
            severity = issue.get("severity", "unknown")
            description = issue.get("missing_info") or issue.get("problem", "No description")
            suggestion = issue.get("suggestion", "")

            feedback_lines.append(f"- [{severity.upper()}] ({issue_type})")
            feedback_lines.append(f"  Problem: {description}")
            if suggestion:
                feedback_lines.append(f"  Suggestion: {suggestion}")
            feedback_lines.append("")

        feedback_lines.append("---")
        feedback_lines.append("")

    return "\n".join(feedback_lines) if feedback_lines else "No specific issues identified."
