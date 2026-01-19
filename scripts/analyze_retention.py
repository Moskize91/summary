"""Analyze retention distribution in snakes."""

from pathlib import Path

from summary.topologization import Topologization


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

    # Load Topologization object
    topo = Topologization(workspace_path)

    # Analyze retention distribution for each snake
    print("\n" + "=" * 80)
    print("Snake Retention Distribution Analysis")
    print("=" * 80)

    # Store percentages for distribution analysis
    retention_percentages = []

    for snake in topo.snake_graph:
        chunks = snake.get_chunks()

        # Count chunks with retention attribute
        chunks_with_retention = sum(1 for chunk in chunks if chunk.retention is not None)
        total_chunks = len(chunks)
        percentage = (chunks_with_retention / total_chunks * 100) if total_chunks > 0 else 0

        retention_percentages.append(percentage)

        print(f"\nSnake {snake.snake_id}: {snake.first_label} → {snake.last_label}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Chunks with retention: {chunks_with_retention}")
        print(f"  Percentage: {percentage:.1f}%")

    # Analyze distribution
    print("\n" + "=" * 80)
    print("Distribution Summary")
    print("=" * 80)

    # Count snakes in different percentage ranges
    zero_percent = sum(1 for p in retention_percentages if p == 0)
    low_percent = sum(1 for p in retention_percentages if 0 < p < 30)
    mid_percent = sum(1 for p in retention_percentages if 30 <= p < 70)
    high_percent = sum(1 for p in retention_percentages if 70 <= p < 100)
    full_percent = sum(1 for p in retention_percentages if p == 100)

    total_snakes = len(retention_percentages)

    print(f"\nTotal snakes: {total_snakes}")
    print(f"  0% retention:    {zero_percent:3d} snakes ({zero_percent/total_snakes*100:.1f}%)")
    print(f"  1-29% retention: {low_percent:3d} snakes ({low_percent/total_snakes*100:.1f}%)")
    print(f"  30-69% retention:{mid_percent:3d} snakes ({mid_percent/total_snakes*100:.1f}%)")
    print(f"  70-99% retention:{high_percent:3d} snakes ({high_percent/total_snakes*100:.1f}%)")
    print(f"  100% retention:  {full_percent:3d} snakes ({full_percent/total_snakes*100:.1f}%)")

    # Check if distribution is concentrated at extremes
    extreme_count = zero_percent + full_percent
    extreme_ratio = extreme_count / total_snakes if total_snakes > 0 else 0

    print(f"\n{'='*80}")
    print(f"Extremes (0% or 100%): {extreme_count} / {total_snakes} = {extreme_ratio*100:.1f}%")

    if extreme_ratio > 0.7:
        print("✓ Distribution is concentrated at extremes (0% and 100%), as expected!")
    elif extreme_ratio > 0.5:
        print("~ Distribution is moderately concentrated at extremes.")
    else:
        print("✗ Distribution is NOT concentrated at extremes.")

    # Close database connection
    topo.close()


if __name__ == "__main__":
    analyze_retention_distribution()
