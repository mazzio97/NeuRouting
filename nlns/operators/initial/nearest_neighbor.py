import numpy as np

from nlns.instances import VRPInstance, VRPSolution, Route


def nearest_neighbor_solution(instance: VRPInstance) -> VRPSolution:
    """Create an initial solution for this instance using a greedy heuristic."""
    solution = [[0]]
    current_load = instance.capacity
    mask = np.array([True] * (instance.n_customers + 1))
    mask[0] = False
    demands = [0] + instance.demands
    while mask.any():
        closest_customer = closest_locations(instance, solution[-1][-1], mask)[0]
        if demands[closest_customer] <= current_load:
            mask[closest_customer] = False
            solution[-1].append(closest_customer)
            current_load -= demands[closest_customer]
        else:
            solution[-1].append(0)
            solution.append([0])
            current_load = instance.capacity
    solution[-1].append(0)
    solution = [Route(r, instance) for r in solution]
    return VRPSolution(instance, solution)


def closest_locations(instance: VRPInstance, location_idx: int, mask=None):
    """Return the idx of the locations sorted by distance."""
    if mask is None:
        mask = np.array([True] * (instance.n_customers + 1))
        mask[location_idx] = False
    distances = np.array(instance.distance_matrix[location_idx])
    distances[~mask] = np.inf
    order = np.argsort(distances)
    last = 0
    for idx in order:
        last += 1
        if distances[idx] is np.inf:
            break
    return order[:last]
