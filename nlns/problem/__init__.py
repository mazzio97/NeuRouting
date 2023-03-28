from __future__ import annotations
from typing import List, Tuple
from itertools import takewhile
from more_itertools import pairwise

import numpy as np

Coord = Tuple[float, float]


class Route(List[int]):
    def __init__(self, nodes: List[Coord]):
        """
        Initialise a problem route by specifying the contained nodes.

        Args:
            nodes (List[int]): Nodes contained by this route.
        """
        super().__init__(nodes)

    @property
    def total_distance(self) -> float:
        """
        Compute the total distance spanned by the route as the
        sum of the euclidean distances between each node.

        Returns:
            float: Total distance.
        """
        return sum(np.linalg.norm(s, e) for s, e in pairwise(self))

    @property
    def complete(self) -> bool:
        """
        Returns:
            bool: True if the route contains more than 1 element and the
                initial and final nodes are the same.
        """
        return len(self) > 1 and (self[0] == self[-1])

    def distance_until_node(self, node: Coord) -> float:
        """
        Compute the cumulative distance from the start of the route
        to a specific node.

        Args:
            node (Coord): Coordinate of the target node.

        Returns:
            float: Distance to the target node.
        """
        assert node in self, f"Node {node} not in this route"
        return sum(np.linalg.norm(s, e)
                   for s, e in pairwise(takewhile(lambda n: n != node, self)))

    def append_route(self, route: Route, position: int = 0):
        """
        Append a route to the current route in the specified position.

        Args:
            route (Route): Route to be added.
            position (int, optional): Position in which the new route
                will be appended. Defaults to 0. Use -1 to append to
                the end of the route.
        """
        assert not self.complete, "Cannot append to a complete route."
        if position == -1:
            position = len(self)

        for idx, el in enumerate(route):
            self.insert(position + idx, el)
