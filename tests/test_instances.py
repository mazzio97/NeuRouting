import pytest
import numpy as np

from context import nlns                                    # NOQA
from helpers import set_default_rng                         # NOQA
from nlns.instances import generate_instances, VRPInstance
from nlns.instances.nazari import (generate_nazari_instances,
                                   generate_nazari_instance)
from nlns.instances.uchoa import generate_uchoa_instances


@pytest.mark.parametrize('n_customers', (20, 50, 100))
def test_generate_nazari_instance(n_customers):
    instance = generate_nazari_instance(n_customers)
    assert len(instance.customers) == n_customers


@pytest.mark.parametrize('seed, np_seed', [(1, None), (None, 1), (1, 2)])
@pytest.mark.parametrize('n_customers', (50,))
def test_generate_nazari_instance_reproducibility(n_customers, seed, np_seed):
    np_rng = None
    np_rng_copy = None
    if np_seed is not None:
        np_rng = np.random.default_rng(np_seed)
        np_rng_copy = np.random.default_rng(np_seed)

    instance = generate_nazari_instance(n_customers, seed=seed, np_rng=np_rng)
    instance_copy = generate_nazari_instance(n_customers, seed=seed,
                                             np_rng=np_rng_copy)

    for customer, customer_copy in zip(instance.customers,
                                       instance_copy.customers):
        assert (customer[0] == customer_copy[0]
                and customer[1] == customer_copy[1])


@pytest.mark.parametrize('distribution', (generate_nazari_instances,
                                          generate_uchoa_instances))
@pytest.mark.parametrize('n_instances', (0, 1, 10, 20))
@pytest.mark.parametrize('n_customers', (20, 50, 100))
def test_generate_distribution_instances(distribution, n_instances,
                                         n_customers):
    instances = distribution(n_instances, n_customers)

    for instance in instances:
        assert isinstance(instance, VRPInstance)
        assert len(instance.customers) == n_customers


@pytest.mark.parametrize('distribution',
                         (generate_nazari_instances,
                          generate_uchoa_instances))
@pytest.mark.parametrize('n_instances', (0, 1, 10, 20))
@pytest.mark.parametrize('n_customers', (20, 50, 100))
def test_generate_distribution_instances_reproducibility(
        distribution, n_instances, n_customers, seed=1):
    instances = distribution(n_instances, n_customers, seed)
    instances_copy = distribution(n_instances, n_customers, seed)

    for instance, instance_copy in zip(instances, instances_copy):
        assert len(instance.customers) == len(instance_copy.customers)
        assert tuple(instance.demands) == tuple(instance_copy.demands)
        assert instance.capacity == instance_copy.capacity


@pytest.mark.parametrize('n_instances, n_customers',
                         [(0, 1), (1, 20), (2, (20, 50)), (10, 10),
                          (10, (20, 50))])
def test_generate_instances(n_instances, n_customers):
    # TODO: test different distributions, test reproducibility
    instances = tuple(generate_instances(n_instances, n_customers))

    assert len(instances) == n_instances

    if type(n_customers) is int:
        n_customers = n_customers,

    instance_customers = [len(instance.customers) for instance in instances]

    for customer in instance_customers:
        assert customer in n_customers

    for customer in n_customers:
        assert not instances or customer in instance_customers
