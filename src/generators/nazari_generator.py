import numpy as np

from instances import VRPInstance

def generate_nazari_instance(n_customers: int) -> VRPInstance:
    """
    Generate data as defined by [1]. The capacity is computed as the linear
    interpolation based on the values picked by [1].

    Args:
        n_customers (int): Number of customer that will be in the generated instance.

    Returns:
        VRPInstance: Final VRPInstance.
    """
    capacity = np.interp(n_customers, [10, 20, 50, 100], [20, 30, 40, 50])
    return VRPInstance(list(np.random.uniform(size=(2,))), 
                       list(np.random.uniform(size=(n_customers, 2))), 
                       list(np.random.randint(1, 10, size=(n_customers,))),
                       capacity)

def generate_nazari_instances(n_instances: int, n_customers: int):
    acceptable = [10, 20, 50, 100]
    assert n_customers in acceptable, f"{n_customers} should be one of {acceptable} for Nazari distribution"
    capacity_map = {10: 20, 20: 30, 50: 40, 100: 50}
    capacity = capacity_map[n_customers]
    return [VRPInstance(tuple(depot), list(customers), list(demands), capacity) for depot, customers, demands
            in zip(list(np.random.uniform(size=(n_instances, 2))),
                   list(np.random.uniform(size=(n_instances, n_customers, 2))),
                   list(np.random.randint(1, 10, size=(n_instances, n_customers))))]
