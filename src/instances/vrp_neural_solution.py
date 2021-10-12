from typing import List

import numpy as np

from instances import VRPInstance
from instances.vrp_solution import VRPSolution, Route


class VRPNeuralSolution(VRPSolution):
    def __init__(self, instance: VRPInstance, routes: List[Route] = None):
        super().__init__(instance, routes)
        self.neural_routes = None
        self.static_repr = None
        self.dynamic_repr = None
        self.map_network_idx_to_route = None
        self.incomplete_nn_idx = None

        self._sync_neural_routes()

    @classmethod
    def from_solution(cls, solution: VRPSolution):
        routes_copy = [Route(route[:], solution.instance) for route in solution.routes]
        return cls(instance=solution.instance, routes=routes_copy)

    def complete_neural_routes(self):
        return [route for route in self.neural_routes if route[0][0] == 0 and route[-1][0] == 0]

    def incomplete_neural_routes(self):
        return [route for route in self.neural_routes if route[0][0] != 0 or route[-1][0] != 0]

    def min_nn_repr_size(self):
        """The neural representation of the solution contains a vector for the depot and one for each node different
        from the depot which is either the end or the beginning of a route."""
        n = 1  # input point for the depot
        for route in self.incomplete_neural_routes():
            if len(route) == 1:
                n += 1
            else:
                if route[0][0] != 0:
                    n += 1
                if route[-1][0] != 0:
                    n += 1
        return n

    def network_representation(self, size):
        min_size = self.min_nn_repr_size()
        assert min_size <= size, f"You should specify an higher size than {size}, the minimum is {min_size}"

        incomplete_tours = self.incomplete_neural_routes()
        instance = self.instance
        nn_input = np.zeros((size, 4))
        nn_input[0, :2] = instance.depot  # Depot location
        nn_input[0, 2] = -1 * instance.capacity  # Depot demand
        nn_input[0, 3] = -1  # Depot state
        nn_idx_to_route = [None] * size
        nn_idx_to_route[0] = [self.neural_routes[0], 0]
        # destroyed_location_idx = []

        i = 1
        for tour in incomplete_tours:
            # Create an input for a tour consisting of a single customer
            if len(tour) == 1:
                nn_input[i, :2] = instance.customers[tour[0][0] - 1]
                nn_input[i, 2] = instance.demands[tour[0][0] - 1]
                nn_input[i, 3] = 1
                tour[0][2] = i
                nn_idx_to_route[i] = [tour, 0]
                # destroyed_location_idx.append(tour[0])
                i += 1
            else:
                # Create an input for the first location in an incomplete tour if the location is not the depot
                if tour[0][0] != 0:
                    nn_input[i, :2] = instance.customers[tour[0][0] - 1]
                    nn_input[i, 2] = sum(c[1] for c in tour)
                    nn_idx_to_route[i] = [tour, 0]
                    tour[0][2] = i
                    if tour[-1][0] == 0:
                        nn_input[i, 3] = 3
                    else:
                        nn_input[i, 3] = 2
                    # destroyed_location_idx.append(tour[0])
                    i += 1
                # Create an input for the last location in an incomplete tour if the location is not the depot
                if tour[-1][0] != 0:
                    nn_input[i, :2] = instance.customers[tour[-1][0] - 1]
                    nn_input[i, 2] = sum(c[1] for c in tour)
                    nn_idx_to_route[i] = [tour, len(tour) - 1]
                    tour[-1][2] = i
                    if tour[0][0] == 0:
                        nn_input[i, 3] = 3
                    else:
                        nn_input[i, 3] = 2
                    # destroyed_location_idx.append(tour[-1])
                    i += 1
        self.incomplete_nn_idx = list(range(1, i))
        self.map_network_idx_to_route = nn_idx_to_route
        self.static_repr = nn_input[:, :2]
        self.dynamic_repr = nn_input[:, 2:]
        return self.static_repr, self.dynamic_repr

    def destroy_nodes(self, to_remove: List[int]):
        super(VRPNeuralSolution, self).destroy_nodes(to_remove)
        self._sync_neural_routes()

    def destroy_edges(self, to_remove: List[tuple]):
        super(VRPNeuralSolution, self).destroy_edges(to_remove)
        self._sync_neural_routes()

    def connect(self, id_from, id_to):
        """Performs an action. The tour end represented by input with the id id_from is connected to the tour end
         presented by the input with id id_to."""
        tour_from = self.map_network_idx_to_route[id_from][0]  # Tour that should be connected
        tour_to = self.map_network_idx_to_route[id_to][0]  # to this tour
        pos_from = self.map_network_idx_to_route[id_from][1]  # Position of the location to connect in tour_from
        pos_to = self.map_network_idx_to_route[id_to][1]  # Position of the location to connect in tour_to

        nn_input_update = []  # Instead of recalculating the tensor representation compute an update

        # Exchange tour_from with tour_to or invert order of the tours.
        # This reduces the number of cases that need to be considered in the following.
        if len(tour_from) > 1 and len(tour_to) > 1:
            if pos_from > 0 and pos_to > 0:
                tour_to.reverse()
            elif pos_from == 0 and pos_to == 0:
                tour_from.reverse()
            elif pos_from == 0 and pos_to > 0:
                tour_from, tour_to = tour_to, tour_from
        elif len(tour_to) > 1:
            if pos_to == 0:
                tour_to.reverse()
            tour_from, tour_to = tour_to, tour_from
        elif len(tour_from) > 1 and pos_from == 0:
            tour_from.reverse()

        # Now we only need to consider two cases 1) Connecting an incomplete tour with more than one location
        # to an incomplete tour with more than one location 2) Connecting an incomplete tour (single
        # or multiple locations) to incomplete tour consisting of a single location

        instance = self.instance
        # Case 1
        if len(tour_from) > 1 and len(tour_to) > 1:
            combined_demand = sum(l[1] for l in tour_from) + sum(l[1] for l in tour_to)
            assert combined_demand <= instance.capacity  # This is ensured by the masking schema

            # The two incomplete tours are combined to one (in)complete tour. All network inputs associated with the
            # two connected tour ends are set to 0
            nn_input_update.append([tour_from[-1][2], 0, 0])
            nn_input_update.append([tour_to[0][2], 0, 0])
            tour_from.extend(tour_to)
            self.neural_routes.remove(tour_to)
            nn_input_update.extend(self._get_network_input_update_for_route(tour_from, combined_demand))

        # Case 2
        if len(tour_to) == 1:
            demand_from = sum(l[1] for l in tour_from)
            combined_demand = demand_from + sum(l[1] for l in tour_to)
            unfulfilled_demand = combined_demand - instance.capacity

            # The new tour has a total demand that is smaller than or equal to the vehicle capacity
            if unfulfilled_demand <= 0:
                if len(tour_from) > 1:
                    nn_input_update.append([tour_from[-1][2], 0, 0])
                # Update solution
                tour_from.extend(tour_to)
                self.neural_routes.remove(tour_to)
                # Generate input update
                nn_input_update.extend(self._get_network_input_update_for_route(tour_from, combined_demand))
            # The new tour has a total demand that is larger than the vehicle capacity
            else:
                nn_input_update.append([tour_from[-1][2], 0, 0])
                if len(tour_from) > 1 and tour_from[0][0] != 0:
                    nn_input_update.append([tour_from[0][2], 0, 0])

                # Update solution
                tour_from.append([tour_to[0][0], tour_to[0][1], tour_to[0][2]])  # deepcopy of tour_to
                tour_from[-1][1] = instance.capacity - demand_from
                tour_from.append([0, 0, 0])
                if tour_from[0][0] != 0:
                    tour_from.insert(0, [0, 0, 0])
                tour_to[0][1] = unfulfilled_demand  # Update demand of tour_to

                nn_input_update.extend(self._get_network_input_update_for_route(tour_to, unfulfilled_demand))

        # Add depot tour to the solution tours if it was removed
        if self.neural_routes[0] != [[0, 0, 0]]:
            self.neural_routes.insert(0, [[0, 0, 0]])
            self.map_network_idx_to_route[0] = [self.neural_routes[0], 0]

        for update in nn_input_update:
            if update[2] == 0 and update[0] != 0:
                self.incomplete_nn_idx.remove(update[0])

        self._sync_default_routes()

        return nn_input_update, tour_from[-1][2]

    def _sync_neural_routes(self):
        demands = [0] + self.instance.demands
        self.neural_routes = [[[c, demands[c], None] if c != 0 else [0, 0, 0] for c in route]
                              for route in [[0]] + self.routes]

    def _sync_default_routes(self):
        self.routes = [Route([c[0] for c in route], self.instance) for route in self.neural_routes[1:]]

    def _get_network_input_update_for_route(self, route, new_demand):
        """Returns an nn_input update for the tour. The demand of the tour is updated to new_demand"""
        nn_input_idx_start = route[0][2]  # Idx of the nn_input for the first location in tour
        nn_input_idx_end = route[-1][2]  # Idx of the nn_input for the last location in tour

        # If the tour stars and ends at the depot, no update is required
        if nn_input_idx_start == 0 and nn_input_idx_end == 0:
            return []

        nn_input_update = []
        # Tour with a single location
        if len(route) == 1:
            if route[0][0] != 0:
                nn_input_update.append([nn_input_idx_end, new_demand, 1])
                self.map_network_idx_to_route[nn_input_idx_end] = [route, 0]
        else:
            # Tour contains the depot
            if route[0][0] == 0 or route[-1][0] == 0:
                # First location in the tour is not the depot
                if route[0][0] != 0:
                    nn_input_update.append([nn_input_idx_start, new_demand, 3])
                    # update first location
                    self.map_network_idx_to_route[nn_input_idx_start] = [route, 0]
                # Last location in the tour is not the depot
                elif route[-1][0] != 0:
                    nn_input_update.append([nn_input_idx_end, new_demand, 3])
                    # update last location
                    self.map_network_idx_to_route[nn_input_idx_end] = [route, len(route) - 1]
            # Tour does not contain the depot
            else:
                # update first and last location of the tour
                nn_input_update.append([nn_input_idx_start, new_demand, 2])
                self.map_network_idx_to_route[nn_input_idx_start] = [route, 0]
                nn_input_update.append([nn_input_idx_end, new_demand, 2])
                self.map_network_idx_to_route[nn_input_idx_end] = [route, len(route) - 1]
        return nn_input_update

    def __deepcopy__(self, memo):
        routes_copy = [Route(route[:], self.instance) for route in self.routes]
        neural_sol_copy = VRPNeuralSolution(self.instance, routes_copy)
        neural_routes_copy = []
        for tour in self.neural_routes:
            neural_routes_copy.append([x[:] for x in tour])
        neural_sol_copy.neural_routes = neural_routes_copy
        return neural_sol_copy
