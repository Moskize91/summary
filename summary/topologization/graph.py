from collections.abc import Iterator
from typing import Generic, TypeVar

NODE = TypeVar("NODE")
EDGE = TypeVar("EDGE")


class Graph(Generic[NODE, EDGE]):
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

    def get_edges(self) -> list[EDGE]:
        """Get all edges in the graph.

        Returns:
            List of Edge objects
        """
        ...  # pylint: disable=unnecessary-ellipsis
