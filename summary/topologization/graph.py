from collections.abc import Iterator
from typing import Generic, TypeVar

NODE = TypeVar("NODE")


class Graph(Generic[NODE]):
    def __iter__(self) -> Iterator[NODE]: ...
