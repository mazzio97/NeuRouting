import numpy as np

from nlns.instances import VRPInstance, VRPSolution, Route

def random_solution(instance: VRPInstance) -> VRPSolution:
    """
    Create a random initial solution for this instance.
    New routes are created and instructed to visit customers 
    in random order. This is repeated until all nodes have been visited.
    A new route is created when an old one reaches the maximum capacity.

    Args:
        instance (VRPInstance): Instance for which a random solution will be generated

    Returns:
        VRPSolution: Random solution
    """
    solution = [[0]]
    current_load = instance.capacity
    mask = np.array([True] * (instance.n_customers + 1))
    mask[0] = False
    demands = [0] + instance.demands
    while mask.any():
        remaining_nodes = mask.nonzero()[0]
        random_node = np.random.choice(remaining_nodes)
        
        if demands[random_node] <= current_load:
            mask[random_node] = False
            solution[-1].append(random_node)
            current_load -= demands[random_node]
        else:
            solution[-1].append(0)
            solution.append([0])
            current_load = instance.capacity
    
    solution[-1].append(0)
    solution = [Route(r, instance) for r in solution]
    return VRPSolution(instance, solution)