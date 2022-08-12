from typing import List, Union, Tuple, Iterable
from itertools import chain

import numpy as np
import torch

from generators.nazari_generator import generate_nazari_instances
from generators.uchoa_generator import generate_uchoa_instances
from instances import VRPInstance


def generate_instance(n_customers: int,
                      distribution: str = 'nazari',
                      seed=42) -> VRPInstance:
    return generate_multiple_instances(1, n_customers, distribution, seed)[0]


def generate_multiple_instances(n_instances: int,
                                n_customers: Union[int, Tuple[int, ...]],
                                distribution: str = 'nazari',
                                seed=42) -> Iterable[VRPInstance]:
    if type(n_customers) is int:
        n_customers = (n_customers,)

    np.random.seed(seed)
    torch.manual_seed(seed)

    assert distribution in ['nazari', 'uchoa'], f"{distribution} is unknown."
    instance_generator = generate_nazari_instances if distribution == 'nazari' \
                         else generate_uchoa_instances
    
    # https://stackoverflow.com/questions/48918627/split-an-integer-into-bins
    instances_per_customer = np.arange(n_instances + len(n_customers) - 1, 
                                       n_instances - 1, -1) // len(n_customers)
    
    return list(chain(*[instance_generator(i, c) for i, c in zip(instances_per_customer, n_customers)]))


    