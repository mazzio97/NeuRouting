from typing import List, Tuple, Dict

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

from nlns.utils.visualize import plot_instance


class VRPInstance:
    """Capacitated VRP instance representation.

    Args:
        depot: Coordinates of the depot.
        customers: Coordinates of the customers. Internally, each
            customer will be identified by its index in this sequence.
        demands: Sequence containing demands for each customer.
        capacity: Carrying capacity of the vehicle(s).
    """

    def __init__(self, depot: Tuple[float, float],
                 customers: List[Tuple[float, float]],
                 demands: List[int], capacity: int):
        assert len(customers) == len(demands)
        self.depot = depot
        self.customers = customers
        self.n_customers = len(customers)
        self.demands = demands
        self.capacity = capacity
        self.distance_matrix = distance_matrix([depot] + customers,
                                               [depot] + customers)

    def adjacency_matrix(self, num_neighbors: int) -> np.array:
        """Generate adjacency matrix according to the works of [1].

        Edge values are classified based on the type of connection they
        provide::

        - 0: node - node
        - 1: node - node knn
        - 2: node self-loop
        - 3: node - depot
        - 4: node - depot knn
        - 5: depot self-loop

        node knn represents the fact that two nodes are k-neighbors of
        each other.

        [1] Kool, W., van Hoof, H., Gromicho, J., Welling, M. (2022). Deep Policy Dynamic Programming for Vehicle Routing Problems

        Args:
            num_neighbors: Number of closest neighbors that each node
                           will be connected to.
        Returns:
            Adjacency matrix
        """                 # NOQA
        num_nodes = self.n_customers
        if num_neighbors == -1:                 # Graph is fully connected
            adj = np.ones((num_nodes + 1, num_nodes + 1))
        else:
            adj = np.zeros((num_nodes + 1, num_nodes + 1))
            # Determine k-nearest neighbors for each node
            knns = np.argpartition(self.distance_matrix, kth=num_neighbors,
                                   axis=-1)[:, num_neighbors::-1]
            # Make connections
            for idx in range(num_nodes):
                adj[idx][knns[idx]] = 1
        np.fill_diagonal(adj, 2)  # Special token for self-connections
        # Special token for depot connection (3 or 4 depending on whether it
        # is knn or not)
        adj[:, 0] += 3
        adj[0, :] += 3
        # Depot self-connection
        adj[0, 0] = 5
        return adj

    def plot(self, ax=None, title=None):
        """Plot instance on matplotlib axis."""
        if ax is None:
            ax = plt.gca()
        plot_instance(ax, self, title)

    def to_networkx(self) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
        """Get networkx graph representation of the instance."""
        coords = [self.depot] + self.customers
        graph = nx.from_numpy_matrix(self.distance_matrix)
        pos = dict(zip(range(self.n_customers + 1), coords))
        return graph, pos
