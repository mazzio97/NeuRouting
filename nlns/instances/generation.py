from typing import Sequence, Union, Iterable, Callable, Optional

import numpy as np

import nlns
from nlns.instances import VRPInstance
from .nazari import generate_nazari_instances
from .uchoa import generate_uchoa_instances

InstancesGenerator = Callable[[int, int, nlns.RandomSeedOrState],
                              'VRPInstance']

distributions = {'nazari': generate_nazari_instances,
                 'uchoa': generate_uchoa_instances}


def generate_instances(n_instances: int,
                       n_customers: Union[int, Sequence[int]],
                       distribution: Union[str, InstancesGenerator] = 'nazari',
                       seed: nlns.RandomSeedOrState = None,
                       preload: Optional[int] = None
                       ) -> Iterable[VRPInstance]:
    """Generate instances from a given distribution.

    By specifying a ``preload`` size it is possible to obtain lazy
    generation.

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
            ``uchoa``.
        seed: A seed or random state for reproducible instances
            generation.
        preload: Any value greater than 0 will determine, at most,
            how many instances will be pregenerated. Tuning this value
            can lead to efficient online generation without filling
            memory.

            In general the greater the value, the faster the generation
            and the higher the memory consumption.

            Default (``None``) leads to pregeneration of as much
            instances as possible. This is the fastest solution for a
            small number of instances.
    """
    if type(n_customers) is int:
        n_customers = n_customers,

    instance_generator = distribution
    if type(distribution) is str:
        instance_generator = distributions[distribution]

    if preload is None:
        preload = n_instances

    # https://stackoverflow.com/questions/48918627/split-an-integer-into-bins
    instances_per_chunk = np.arange(n_instances + len(n_customers) - 1,
                                    n_instances - 1, -1) // len(n_customers)

    rng = nlns.get_rng(seed)

    if n_instances <= 0:
        return

    for chunk_instances, customers in zip(instances_per_chunk, n_customers):
        # Divide per customer chunks into subchunks, eventually based
        # on preload size
        n_to_generate = min(chunk_instances, preload)

        # Compute the total number of subchunks and an eventual
        # remaining
        full_preloads, partial_preload = divmod(chunk_instances, n_to_generate)
        for _ in range(full_preloads):
            yield from instance_generator(n_to_generate, customers,
                                          rng.random())

        if partial_preload > 0:
            yield from instance_generator(partial_preload, customers,
                                          rng.random())


def generate_instance(n_customers: int,
                      distribution: Union[str, InstancesGenerator] = 'nazari',
                      seed: nlns.RandomSeedOrState = None) -> VRPInstance:
    """Generate an instance from a given distribution.

    See :func:`generate_instances`.
    """
    return next(generate_instances(1, n_customers, distribution, seed))
