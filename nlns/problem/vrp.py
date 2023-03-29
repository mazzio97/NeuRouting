from typing import List, Dict
from itertools import takewhile
from more_itertools import pairwise

import numpy as np
from scipy.spatial import distance_matrix

from nlns.problem import Coord, Route


class VRPInstance:
    def __init__(self, depot: Coord, nodes: List[Coord], demands: List[int]):
        """
        A Vehicle Routing Problem instance composed of the depot position,
        nodes positions and the requirement of each node.

        Args:
            depot (Coord): X and Y coordinate of the depot. Every vehicle route
                starts and ends at the depot.
            nodes (List[Coord]): Coordinates of the nodes
            demands (List[int]):
                Demands of each nodes, expressed as an integer.
        """
        assert len(nodes) == len(demands)
        self.depot = depot
        self.nodes = nodes
        self.demands = demands
        self.distance_matrix = distance_matrix([depot] + nodes,
                                               [depot] + nodes)

    def adjacency_matrix(self,
                         num_neighbors: int = -1,
                         edge_encoding: Dict[str, int] = {
                            "node-node": 1,
                            "node-node_knn": 2,
                            "node-self": 3,
                            "node-depot": 4,
                            "node-depot_knn": 5,
                            "depot-self": 6, }) -> np.array:
        """
        Generate adjacency matrix. The encoding is based on the work of [1].
        Edge values are classified based on the type of connection
        they provide and can be customised using the edge_encoding parameter.

        [1] Kool, W., van Hoof, H., Gromicho, J., Welling, M. (2022).
            Deep Policy Dynamic Programming for Vehicle Routing Problems

        Args:
            num_neighbors (int, optional): Number of closest neighbours that
                each node will be connected to.
                Defaults to -1: fully connected graph.
            edge_encoding (Dict[str, int], optional):
                Encoding value for each edge type.
                The types are:
                    - "node-node": Edge between two general nodes;
                    - "node-node_knn": Edge between two general nodes when
                        they are neighbours;
                    - "node-self": Self connection edge;
                    - "node-depot": Connection between a general node
                        and the depot;
                    - "node-depot_knn": Connection between a general node
                        and the depot when they are
                        neighbours;
                    - "depot-self": Self loop of the depot.
                Defaults to the encoding defined by [1].
        Returns:
            np.array: Adjacency matrix
        """
        num_nodes = len(self.nodes)
        adj = np.zeros((num_nodes + 1, num_nodes + 1))

        if num_neighbors == -1:
            num_neighbors = num_nodes

        for k in ["node-node", "node-node_knn", "node-self", "node-depot",
                  "node-depot_knn", "depot-self"]:
            assert k in edge_encoding, \
                    f"'{k}'' key is missing from edge_encoding"

        # Determine k-nearest neighbors for each node
        knns = np.argpartition(
            self.distance_matrix,
            kth=num_neighbors,
            axis=-1)[:, num_neighbors::-1]

        # Make connections
        for idx in range(1, num_nodes + 1):
            # Set knn connection
            adj[idx][knns[idx]] = edge_encoding["node-node_knn"]
            # Set connection with the depot depending if knn
            adj[idx, 0] = edge_encoding["node-depot_knn"] \
                if 0 in knns[idx] else  \
                edge_encoding["node-depot"]

        # Set self-connections
        np.fill_diagonal(adj, edge_encoding["node-self"])

        # Depot self-connection
        adj[0, 0] = edge_encoding["depot-self"]
        return adj


class VRPRoute(Route):
    def __init__(self, nodes: List[Coord], demands: List[int]):
        """
        Initialise a problem route by specifying the nodes
        positions and their respective demands.

        Args:
            nodes (List[int]): Nodes contained by this route.
            demands (List[int]): Demands of each node in the route.
        """
        super().__init__(nodes)
        self.demands = demands

    @property
    def total_demand(self) -> int:
        """
        Returns:
            int: The total demand of this route.
        """
        return sum(self.demands)

    def demand_until_customer(self, node_coord: Coord) -> float:
        """
        Compute the cumulative demand from the start of the route
        to a specific node.

        Args:
            node_coord (Coord): Coordinate of the target nodes.

        Returns:
            float: Distance to the target node.
        """
        assert node_coord in self, f"Node at {node_coord} not in this route"
        return sum(np.linalg.norm(s, e)
                   for s, e in
                   pairwise(takewhile(lambda c: c[0] != node_coord,
                                      zip(self, self.demands))))
