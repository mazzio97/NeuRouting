from typing import Optional, Tuple

import numpy as np

import nlns
from nlns.instances import VRPInstance


def generate_nazari_instances(instances: int, n_customers: int,
                              seed: nlns.RandomSeedOrState = None
                              ) -> Tuple[VRPInstance, ...]:
    """Sample from :func:`generate_nazari_instance` multiple times.

    Args:
        n_customers: Number of customer that will be in the generated instance.
        seed: A seed or random state for reproducible results.
    """
    rng = nlns.numpy_generator_from_rng(nlns.get_rng(seed))

    return tuple(generate_nazari_instance(n_customers, np_rng=rng)
                 for _ in range(instances))


def generate_nazari_instance(n_customers: int,
                             seed: nlns.RandomSeedOrState = None,
                             np_rng: Optional[np.random.Generator] = None
                             ) -> VRPInstance:
    """Generate a VRP Instance based on Nazari[1] distribution.

    Capacity follows the work of Kool[2] in which a node demand
    is expressed as a fraction of a whole capacity.
    The capacity value is computed as the linear interpolation
    between the values presented in the paper itself.

    For reproducibility, provide

    [1] https://proceedings.neurips.cc/paper/2018/hash/9fb4651c05b2ed70fba5afe0b039a550-Abstract.html

    [2] https://arxiv.org/pdf/1803.08475.pdf

    Args:
        n_customers: Number of customer that will be in the generated instance.
        seed: A seed or random state for reproducible results.
        np_rng: A numpy random number generator for reproducible
            results. If given, it overrides the ``seed`` parameter.
            This option is faster as the calculations are numpy based
            and it does not require construction of a new generator.

    Returns:
        VRPInstance: Final VRPInstance.
    """                 # NOQA
    if np_rng is not None:
        rng = np_rng
    else:
        rng = nlns.numpy_generator_from_rng(nlns.get_rng(seed))

    capacity = np.interp(n_customers, [10, 20, 50, 100], [20, 30, 40, 50])
    return VRPInstance(list(rng.uniform(size=(2,))),
                       list(rng.uniform(size=(n_customers, 2))),
                       list(rng.integers(1, 10, size=(n_customers,))),
                       capacity)
