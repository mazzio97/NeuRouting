from typing import Sequence, Union, Iterable, Callable
from itertools import chain

import numpy as np

import nlns
from nlns.generators.nazari_generator import generate_nazari_instances
from nlns.generators.uchoa_generator import generate_uchoa_instances
from nlns.instances import VRPInstance

InstancesGenerator = Callable[[int, int, nlns.RandomSeedOrState], VRPInstance]

distributions = {'nazari': generate_nazari_instances,
                 'uchoha': generate_uchoa_instances}


def generate_instances(n_instances: int,
                       n_customers: Union[int, Sequence[int]],
                       distribution: Union[str, InstancesGenerator] = 'nazari',
                       seed: nlns.RandomSeedOrState = None
                       ) -> Iterable[VRPInstance]:
    """Generate instances from a given distribution.

    Args:
        n_instances: Total number of instances to generate.
        n_customers: Either an int (number of customers for all
            generated instance) or a sequence of ints. In the latter
            case, generation is split into ``len(n_customers)`` chunks,
            each of which consisting in instances having the
            corresponding number of customers.
        distribution: The generator to be used for instance generation.
            Either a function in the format specified by
            :attr:`InstancesGenerator` or a string representing a known
            distributions. Known distributions are ``nazari`` and
            ``uchoha``.
        seed: A seed or random state for reproducible instances
            generation.
    """
    if type(n_customers) is int:
        n_customers = n_customers,

    instance_generator = distribution
    if type(distribution) is str:
        instance_generator = distributions[distribution]

    # https://stackoverflow.com/questions/48918627/split-an-integer-into-bins
    instances_per_customer = np.arange(n_instances + len(n_customers) - 1,
                                       n_instances - 1, -1) // len(n_customers)

    rng = nlns.get_rng(seed)
    seeds_per_customer = tuple(rng.random() for _ in n_customers)

    return (chain(*(instance_generator(i, c, s)
                    for i, c, s in zip(instances_per_customer, n_customers,
                                       seeds_per_customer))))


def generate_instance(n_customers: int,
                      distribution: Union[str, InstancesGenerator] = 'nazari',
                      seed: nlns.RandomSeedOrState = None) -> VRPInstance:
    """Generate an instance from a given distribution.

    See :func:`generate_instances`.
    """
    return next(generate_instances(1, n_customers, distribution, seed))
