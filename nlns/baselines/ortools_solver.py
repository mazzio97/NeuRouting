from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from nlns.environments import VRPSolver
from nlns.instances import VRPSolution, VRPInstance
from nlns.instances.vrp_solution import Route
from nlns.utils.vrp_io import read_vrp, GRID_DIM


class ORToolsSolver(VRPSolver):
    def __init__(self):
        super().__init__("ortools")
        self._data = None
        self._manager = None
        self._routing = None

    def reset(self, instance: VRPInstance):
        self.instance = instance
        n_vehicles = instance.n_customers  # int(sum(instance.demands) // instance.capacity) + 2
        self._data = {
            'distance_matrix': instance.distance_matrix * GRID_DIM,
            'demands': [0] + instance.demands,
            'num_vehicles': n_vehicles,
            'vehicle_capacities': [instance.capacity] * n_vehicles,
            'depot': 0
        }
        self._manager = pywrapcp.RoutingIndexManager(instance.n_customers + 1, n_vehicles, self._data['depot'])
        self._routing = pywrapcp.RoutingModel(self._manager)
        transit_callback_index = self._routing.RegisterTransitCallback(self._distance_callback)
        self._routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        demand_callback_index = self._routing.RegisterUnaryTransitCallback(self._demand_callback)
        self._routing.AddDimensionWithVehicleCapacity(evaluator_index=demand_callback_index,
                                                      slack_max=0,  # null capacity slack
                                                      vehicle_capacities=self._data['vehicle_capacities'],
                                                      fix_start_cumul_to_zero=True,  # start cumulative to zero
                                                      name='Capacity')

    def solve(self, instance: VRPInstance, time_limit=None, max_steps=None) -> VRPSolution:
        self.reset(instance)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        if time_limit is not None:
            search_parameters.time_limit.FromSeconds(time_limit)
        if max_steps is not None:
            search_parameters.solution_limit = max_steps
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

        solution = self._routing.SolveWithParameters(search_parameters)
        self.solution = self._process_solution(solution)
        return self.solution

    def _distance_callback(self, from_index: int, to_index: int) -> float:
        """Returns the distance between the two nodes."""
        from_node = self._manager.IndexToNode(from_index)
        to_node = self._manager.IndexToNode(to_index)
        return self._data['distance_matrix'][from_node][to_node]

    def _demand_callback(self, from_index: int) -> int:
        """Returns the demand of the node."""
        from_node = self._manager.IndexToNode(from_index)
        return self._data['demands'][from_node]

    def _process_solution(self, solution) -> VRPSolution:
        routes = []
        for vehicle_id in range(self._data['num_vehicles']):
            index = self._routing.Start(vehicle_id)
            path = []
            while not self._routing.IsEnd(index):
                node_index = self._manager.IndexToNode(index)
                path.append(node_index)
                index = solution.Value(self._routing.NextVar(index))
            path.append(self._manager.IndexToNode(index))
            # Check if a vehicle remained in the depot
            if len(path) > 2:
                routes.append(path)
        routes = [Route(r, self.instance) for r in routes]
        return VRPSolution(self.instance, routes)


if __name__ == "__main__":
    inst = read_vrp("../../res/A-n32-k5.vrp", grid_dim=100)
    or_solver = ORToolsSolver()
    or_solver.solve(inst, time_limit=10)
    or_solver.render()
