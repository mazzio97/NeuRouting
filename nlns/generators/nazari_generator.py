import numpy as np

from nlns.instances import VRPInstance

def generate_nazari_instances(x, y): pass

def generate_nazari_instance(n_customers: int) -> VRPInstance:
    """
    Generate a VRP Instance based on Nazari[1] distribution.
    Capacity follows the work of Kool[2] in which a node demand
    is expressed as a fraction of a whole capacity.
    The capacity value is computed as the linear interpolation
    between the values presented in the paper itself.

    [1] https://proceedings.neurips.cc/paper/2018/hash/9fb4651c05b2ed70fba5afe0b039a550-Abstract.html
    [2] https://arxiv.org/pdf/1803.08475.pdf

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