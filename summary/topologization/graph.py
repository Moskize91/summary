from collections.abc import Iterator
from typing import Generic, TypeVar

NODE = TypeVar("NODE")


class Graph(Generic[NODE]):
    """Graph protocol for knowledge graph and snake graph.

    Provides iteration over nodes and access to edges.
    Implementations should provide efficient lazy-loading.
    """

    def __iter__(self) -> Iterator[NODE]:
        """Iterate over all nodes in the graph.

        Returns:
            Iterator over nodes
        """
        ...  # pylint: disable=unnecessary-ellipsis

    def get_edges(self) -> list[tuple[int, ...]]:
        """Get all edges in the graph.

        Returns:
            List of edge tuples. Format depends on implementation:
            - KnowledgeGraph: [(from_id, to_id), ...]
            - SnakeGraph: [(from_id, to_id, count), ...]
        """
        ...  # pylint: disable=unnecessary-ellipsis
