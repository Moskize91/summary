"""Analyze retention distribution in snakes."""

from pathlib import Path

from summary.topologization import ReadonlyTopologization


def analyze_retention_distribution():
    """Analyze retention attribute distribution across snakes."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    workspace_path = project_root / "workspace"

    # Check if workspace exists
    if not workspace_path.exists():
        print(f"Error: Workspace not found at {workspace_path}")
        print("Please run the main extraction process first (python main.py)")
        return

    print(f"Loading workspace from: {workspace_path}")

    # Load ReadonlyTopologization object
    topo = ReadonlyTopologization(workspace_path)

    # Get all chapters
    chapter_ids = topo.get_all_chapter_ids()
    print(f"\nFound {len(chapter_ids)} chapter(s): {chapter_ids}")

    # Store percentages for global distribution analysis
    all_retention_percentages = []
    total_snakes_count = 0

    # Process each chapter and group
    for chapter_id in chapter_ids:
        group_ids = topo.get_group_ids_for_chapter(chapter_id)
        print(f"\nChapter {chapter_id}: {len(group_ids)} group(s)")

        for group_id in group_ids:
            print("\n" + "=" * 80)
            print(f"Chapter {chapter_id}, Group {group_id} - Snake Retention Distribution")
            print("=" * 80)

            # Get snake graph for this chapter/group
            snake_graph = topo.get_snake_graph(chapter_id, group_id)

            # Store percentages for this group
            group_retention_percentages = []

            for snake in snake_graph:
                chunks = snake.get_chunks()

                # Count chunks with retention attribute
                chunks_with_retention = sum(1 for chunk in chunks if chunk.retention is not None)
                total_chunks = len(chunks)
                percentage = (chunks_with_retention / total_chunks * 100) if total_chunks > 0 else 0

                group_retention_percentages.append(percentage)
                all_retention_percentages.append(percentage)

                print(f"\nSnake {snake.snake_id}: {snake.first_label} → {snake.last_label}")
                print(f"  Total chunks: {total_chunks}")
                print(f"  Chunks with retention: {chunks_with_retention}")
                print(f"  Percentage: {percentage:.1f}%")

            # Analyze distribution for this group
            if group_retention_percentages:
                print(f"\n{'-' * 80}")
                print(f"Group {group_id} Distribution Summary")
                print(f"{'-' * 80}")

                zero_percent = sum(1 for p in group_retention_percentages if p == 0)
                low_percent = sum(1 for p in group_retention_percentages if 0 < p < 30)
                mid_percent = sum(1 for p in group_retention_percentages if 30 <= p < 70)
                high_percent = sum(1 for p in group_retention_percentages if 70 <= p < 100)
                full_percent = sum(1 for p in group_retention_percentages if p == 100)

                group_total = len(group_retention_percentages)
                total_snakes_count += group_total

                print(f"\nGroup snakes: {group_total}")
                print(f"  0% retention:    {zero_percent:3d} snakes ({zero_percent / group_total * 100:.1f}%)")
                print(f"  1-29% retention: {low_percent:3d} snakes ({low_percent / group_total * 100:.1f}%)")
                print(f"  30-69% retention:{mid_percent:3d} snakes ({mid_percent / group_total * 100:.1f}%)")
                print(f"  70-99% retention:{high_percent:3d} snakes ({high_percent / group_total * 100:.1f}%)")
                print(f"  100% retention:  {full_percent:3d} snakes ({full_percent / group_total * 100:.1f}%)")

    # Global distribution summary
    if all_retention_percentages:
        print("\n" + "=" * 80)
        print("GLOBAL Distribution Summary (All Chapters & Groups)")
        print("=" * 80)

        zero_percent = sum(1 for p in all_retention_percentages if p == 0)
        low_percent = sum(1 for p in all_retention_percentages if 0 < p < 30)
        mid_percent = sum(1 for p in all_retention_percentages if 30 <= p < 70)
        high_percent = sum(1 for p in all_retention_percentages if 70 <= p < 100)
        full_percent = sum(1 for p in all_retention_percentages if p == 100)

        print(f"\nTotal snakes across all groups: {total_snakes_count}")
        print(f"  0% retention:    {zero_percent:3d} snakes ({zero_percent / total_snakes_count * 100:.1f}%)")
        print(f"  1-29% retention: {low_percent:3d} snakes ({low_percent / total_snakes_count * 100:.1f}%)")
        print(f"  30-69% retention:{mid_percent:3d} snakes ({mid_percent / total_snakes_count * 100:.1f}%)")
        print(f"  70-99% retention:{high_percent:3d} snakes ({high_percent / total_snakes_count * 100:.1f}%)")
        print(f"  100% retention:  {full_percent:3d} snakes ({full_percent / total_snakes_count * 100:.1f}%)")

        # Check if distribution is concentrated at extremes
        extreme_count = zero_percent + full_percent
        extreme_ratio = extreme_count / total_snakes_count if total_snakes_count > 0 else 0

        print(f"\n{'=' * 80}")
        print(f"Extremes (0% or 100%): {extreme_count} / {total_snakes_count} = {extreme_ratio * 100:.1f}%")

        if extreme_ratio > 0.7:
            print("✓ Distribution is concentrated at extremes (0% and 100%), as expected!")
        elif extreme_ratio > 0.5:
            print("~ Distribution is moderately concentrated at extremes.")
        else:
            print("✗ Distribution is NOT concentrated at extremes.")
    else:
        print("\nNo snakes found in workspace.")

    # Close database connection
    topo.close()


if __name__ == "__main__":
    analyze_retention_distribution()
