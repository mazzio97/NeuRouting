from typing import List, Tuple, Iterable

import numpy as np
from matplotlib import pyplot as plt
from more_itertools import split_after

from nlns.instances.vrp_instance import VRPInstance
from nlns.utils.visualize import plot_solution


class Route(List[int]):
    """VRP route, associated to a specific instance.

    Represented by: a list of customer ids (ints).

    Args:
        nodes: Iterable of customer ids representing the route.
        instance: Associated VRP instance.
    """

    def __init__(self, nodes: Iterable[int], instance: VRPInstance):
        super().__init__(nodes)
        self.distance_matrix = instance.distance_matrix
        self.demands = instance.demands

    def is_complete(self) -> bool:
        """Whether the route is complete (starts and ends at depot)."""
        return len(self) > 1 and (self[0] == self[-1] == 0)

    def is_incomplete(self) -> bool:
        """Whether the route is not complete.

        Semantically equivalent to not :attr:`is_complete`.
        """
        return self[0] != 0 or self[-1] != 0

    def total_distance(self) -> float:
        """Total distance (cost) convered by the route."""
        return sum(self.distance_matrix[from_idx, to_idx]
                   for from_idx, to_idx in zip(self[:-1], self[1:]))

    def distance_till_customer(self, customer_idx: int) -> float:
        """Total distance from start to specific customer."""
        assert customer_idx in self, (
            f'Customer {customer_idx} not in this route')
        distance = 0
        for from_idx, to_idx in zip(self[:-1], self[1:]):
            distance += self.distance_matrix[from_idx, to_idx]
            if to_idx == customer_idx:
                break
        return distance

    def total_demand(self) -> int:
        """Total demand of customers in the route."""
        demands = [0] + self.demands
        return sum(demands[idx] for idx in self)

    def demand_till_customer(self, customer_idx: int) -> int:
        """Total demand of customers from start to specific customer."""
        assert customer_idx in self, (
            f'Customer {customer_idx} not in this route')
        demands = [0] + self.demands
        demand = 0
        for idx in self:
            demand += demands[idx]
            if idx == customer_idx:
                break
        return demand

    def append_route(self, route: List[int], self_begin: bool,
                     route_begin: bool):
        """Append a route to self.

        Args:
            route: The route to be appended.
            self_begin: If True, append the given route after the end
                of self.
            route_begin: If True, insert the given route before the
                start of self.
        """
        assert self.is_incomplete(), 'Cannot append to a complete route.'
        if not route_begin:
            route.reverse()
        if not self_begin:
            for el in route:
                self.append(el)
        else:
            for el in route:
                self.insert(0, el)


class VRPSolution:

    def __init__(self, instance: VRPInstance, routes: List[Route] = None):
        self.instance = instance
        self.routes = routes if routes is not None else []
        self.time_taken = -1

    @classmethod
    def from_edges(cls, instance: VRPInstance, edges: List[tuple] = None):
        routes = []
        for edge in edges:
            if edge[0] == 0:
                successor = edge[1]
                path = [0, successor]
                while successor != 0:
                    for e in edges:
                        if e[0] == successor:
                            successor = e[1]
                            path.append(successor)
                path = Route(path, instance)
                routes.append(path)
        return cls(instance=instance, routes=routes)

    def as_edges(self) -> List[tuple]:
        return [(from_id, to_id)
                for route in self.complete_routes() + self.incomplete_routes()
                for from_id, to_id in zip(route[:-1], route[1:])]

    def adjacency_matrix(self) -> np.ndarray:
        adj = np.zeros((self.instance.n_customers + 1,
                        self.instance.n_customers + 1), dtype=int)
        for i, j in self.as_edges():
            # nodes_target[i] = idx  # node targets: ordering of nodes in tour
            adj[i][j] = 1
            adj[j][i] = 1
        return adj

    def isolated_customers(self) -> List[int]:
        return [route[0] for route in self.routes if len(route) == 1]

    def missing_customers(self) -> List[int]:
        missing = set()
        for route in self.routes:
            if route[0] != 0:
                missing.add(route[0])
            if route[-1] != 0:
                missing.add(route[-1])
        return list(missing)

    def verify(self) -> bool:
        # Each tour does not exceed the vehicle capacity
        demands = [0] + self.instance.demands
        loads = [sum(demands[node] for node in tour) for tour in self.routes]
        for i, load in enumerate(loads):
            assert load <= self.instance.capacity, (
                f'Tour {i} with a total of {load} exceeds the maximum '
                f'capacity of {self.instance.capacity}.')
        # Each tour starts and ends at the depot
        for i, route in enumerate(self.routes):
            assert route[0] == route[-1] == 0, (
                f'Route {i} is incomplete because it starts at {route[0]} '
                f'and ends at {route[-1]}.')
        # All customers have been visited
        missing = self.missing_customers()
        assert len(missing) == 0, \
            f"Customers {missing} are not present in any tour."
        return True

    @property
    def cost(self) -> float:
        return sum(self.instance.distance_matrix[from_id, to_id]
                   for from_id, to_id in self.as_edges())

    def complete_routes(self) -> List[Route]:
        return [route for route in self.routes if route.is_complete()]

    def incomplete_routes(self) -> List[Route]:
        return [route for route in self.routes
                if route.is_incomplete() and len(route) > 1]

    def get_customer_route(self, customer_idx: int) -> Route:
        for route in self.routes:
            if customer_idx in route:
                return route

    def get_edge_route(self, edge: Tuple[int, int]) -> Route:
        for route in self.routes:
            if edge[0] in route and edge[1] in route:
                return route

    def destroy_nodes(self, to_remove: List[int]):
        incomplete_routes = []
        complete_routes = []
        removed = []
        for route in self.routes:
            last_split_idx = 0
            for i in range(1, len(route) - 1):
                if route[i] in to_remove:
                    # Create two new tours:
                    # The first consisting of the tour from the depot or from
                    # the last removed customer to the customer that should
                    # be removed
                    if i > last_split_idx and i > 1:
                        new_tour_pre = route[last_split_idx:i]
                        # complete_routes.append(new_tour_pre)
                        incomplete_routes.append(new_tour_pre)
                    # The second consisting of only the customer to be removed
                    customer_idx = route[i]
                    # make sure the customer has not already been extracted
                    # from a different tour
                    if customer_idx not in removed:
                        new_tour = [customer_idx]
                        incomplete_routes.append(new_tour)
                        removed.append(customer_idx)
                    last_split_idx = i + 1
            if last_split_idx > 0:
                # Create another new tour consisting of the remaining part
                # of the original tour
                if last_split_idx < len(route) - 1:
                    new_tour_post = route[last_split_idx:]
                    incomplete_routes.append(new_tour_post)
            else:  # add unchanged tour
                complete_routes.append(route)
        complete_routes = [Route(cr, self.instance)
                           for cr in complete_routes]
        incomplete_routes = [Route(ir, self.instance)
                             for ir in incomplete_routes]
        self.routes = complete_routes + incomplete_routes

    def destroy_edges(self, to_remove: List[tuple]):
        for edge in to_remove:
            route = self.get_edge_route(edge)
            splits = list(split_after(route, lambda x: x == edge[0]))
            assert len(splits) == 2, f"{route}: {splits} because of {edge}"
            presplit, postsplit = splits
            self.routes.remove(route)
            if presplit != [0]:
                presplit = Route(presplit, self.instance)
                self.routes.append(presplit)
            if postsplit != [0]:
                postsplit = Route(postsplit, self.instance)
                self.routes.append(postsplit)

    def plot(self, ax=None, title=None):
        if ax is None:
            ax = plt.gca()
        plot_solution(ax, self, title)

    def __deepcopy__(self, memo):
        routes_copy = [Route(route[:], self.instance) for route in self.routes]
        return VRPSolution(self.instance, routes_copy)
