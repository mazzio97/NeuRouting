"""VRP instance generation as defined by Uchoa et al. (2017)[1]

Use :func:`generate_uchoa_instances` to sample instances as desired.
Other functions are mostly for internal use. For a more flexible
interface see :func:`nlns.instances.generate_instances`.

[1] Uchoa, Eduardo & Pecin, Diego & Pessoa, Artur & Poggi, Marcus & Vidal, Thibaut & Subramanian, Anand. (2016). New Benchmark Instances for the Capacitated Vehicle Routing Problem. European Journal of Operational Research. 257. 10.1016/j.ejor.2016.08.012.
"""             # NOQA
import math

import numpy as np

import nlns
from nlns.instances import VRPInstance

GRID_SIZE = 1000


def generate_depot_coordinates(n_instances, depot_type=None,
                               np_rng: np.random.Generator = None):
    if np_rng is None:
        np_rng = np.random.default_rng()

    # Depot Position
    # 0 = central (500, 500), 1 = eccentric (0, 0), 2 = random
    depot_types = np_rng.integers(0, 3, size=n_instances)
    if depot_type is not None:  # mix
        # Central, Eccentric, Random
        codes = {'C': 0, 'E': 1, 'R': 2}
        depot_types[:] = codes[depot_type.upper()]

    depot_locations = np_rng.uniform(0, GRID_SIZE, size=(n_instances, 2))
    depot_locations[depot_types == 0] = GRID_SIZE / 2
    depot_locations[depot_types == 1] = 0
    return depot_locations, depot_types


def generate_clustered(num_seeds, num_samples, max_seeds=None,
                       np_rng: np.random.Generator = None):
    if np_rng is None:
        np_rng = np.random.default_rng()

    if max_seeds is None:
        max_seeds = num_seeds.max().item()
    n_instances = num_seeds.shape[0]
    batch_rng = np.arange(n_instances, dtype=np.int64)
    seed_coords = np_rng.uniform(0, GRID_SIZE,
                                 size=(n_instances, max_seeds, 2))
    # We make a little extra since some may fall off the grid
    n_try = num_samples * 2
    while True:
        loc_seed_ind = (np_rng.random(size=(n_instances, n_try))
                        * num_seeds[:, None]).astype(np.int64)

        loc_seeds = seed_coords[batch_rng[:, None], loc_seed_ind]
        alpha = np_rng.uniform(0, 2 * math.pi, size=(n_instances, n_try))
        d = np.log(np_rng.random(size=(n_instances, n_try))) * -40
        coords = (np.stack((np.sin(alpha), np.cos(alpha)), -1)
                  * d[:, :, None] + loc_seeds)

        feas = ((coords >= 0) & (coords <= GRID_SIZE)).sum(-1) == 2
        ind_topk = np.argpartition(feas, -num_samples, axis=-1).take(
            indices=range(-num_samples, 0), axis=-1)
        if np.take_along_axis(feas, ind_topk, axis=-1).all():
            break
        n_try *= 2  # Increase if this fails
    return coords[batch_rng[:, None], ind_topk]


def generate_customer_coordinates(n_instances, n_customers, min_seeds=3,
                                  max_seeds=8, customer_type=None,
                                  np_rng: np.random.Generator = None):
    if np_rng is None:
        np_rng = np.random.default_rng()

    # Customer position
    # 0 = random, 1 = clustered, 2 = random clustered 50/50
    # We always do this so we always pull the same number of random numbers
    customer_types = np_rng.integers(0, 3, size=n_instances)
    if customer_type is not None:  # Mix
        # Random, Clustered, Random-Clustered (half half)
        codes = {'R': 0, 'C': 1, 'RC': 2}
        customer_types[:] = codes[customer_type.upper()]

    # Sample number of seeds uniform (inclusive)
    num_seeds = np_rng.integers(min_seeds, max_seeds, endpoint=True,
                                size=n_instances)

    # We sample random and clustered coordinates for all instances,
    # this way, the instances in the 'mix' case
    # Will be exactly the same as the instances in one of the tree
    # 'not mixed' cases and we can reuse evaluations
    rand_coords = np_rng.uniform(0, GRID_SIZE,
                                 size=(n_instances, n_customers, 2))
    clustered_coords = generate_clustered(num_seeds, n_customers,
                                          max_seeds=max_seeds,
                                          np_rng=np_rng)

    # Clustered
    rand_coords[customer_types == 1] = clustered_coords[customer_types == 1]
    # Half clustered
    rand_coords[customer_types == 2, :(n_customers // 2)] = \
        clustered_coords[customer_types == 2, :(n_customers // 2)]

    return rand_coords, customer_types


def generate_demands(customers, np_rng: np.random.Generator = None):
    if np_rng is None:
        np_rng = np.random.default_rng()

    n_instances, n_customers, _ = customers.shape
    # Demand distribution
    # 0 = unitary (1)
    # 1 = small values, large variance (1-10)
    # 2 = small values, small variance (5-10)
    # 3 = large values, large variance (1-100)
    # 4 = large values, large variance (50-100)
    # 5 = depending on quadrant top left and bottom right
    #     (even quadrants) (1-50), others (51-100) so add 50
    # 6 = many small, few large most (70 to 95 %, unclear so take
    #     uniform) from (1-10), rest from (50-100)
    lb = np.array([1, 1, 5, 1, 50, 1, 1], dtype=np.int64)
    ub = np.array([1, 10, 10, 100, 100, 50, 10], dtype=np.int64)
    customer_positions = np_rng.integers(0, 7, size=n_instances)
    lb_ = lb[customer_positions, None]
    ub_ = ub[customer_positions, None]
    # Make sure we always sample the same number of random numbers
    rand_1 = np_rng.random(size=(n_instances, n_customers))
    rand_2 = np_rng.random(size=(n_instances, n_customers))
    rand_3 = np_rng.random(size=n_instances)
    demands = (rand_1 * (ub_ - lb_ + 1)).astype(np.int64) + lb_
    # either both smaller than grid_size // 2 results in 2 inequalities
    # satisfied, or both larger 0
    # in all cases it is 1 (odd quadrant) and we should add 50

    demands[customer_positions == 5] += (
        (customers[customer_positions == 5]
         < GRID_SIZE // 2).sum(-1) == 1) * 50
    # slightly different than in the paper we do not exactly pick a
    # value between 70 and 95 % to have a large value
    # but based on the threshold we let each individual location have a
    # large demand with this probability
    demands_small = demands[customer_positions == 6]
    demands[customer_positions == 6] = np.where(
        rand_2[customer_positions == 6] > (rand_3 * 0.25 + 0.70)
              [customer_positions == 6, None],
        demands_small,
        (rand_1[customer_positions == 6] * (100 - 50 + 1))
        .astype(np.int64) + 50
    )
    return demands


def generate_uchoa_instances(n_instances: int, n_customers: int,
                             seed: nlns.RandomSeedOrState = None):
    if n_instances <= 0:
        return []

    np_rng = nlns.numpy_generator_from_rng(nlns.get_rng(seed))

    depot, depot_types = generate_depot_coordinates(n_instances, np_rng=np_rng)
    customers, customer_types = generate_customer_coordinates(n_instances,
                                                              n_customers,
                                                              np_rng=np_rng)
    demands = generate_demands(customers, np_rng)
    r = np_rng.triangular(3, 6, 25, size=n_instances)
    capacities = np.ceil(r * demands.mean(-1))
    # It can happen that demand is larger than capacity, so cap demand
    exp_capacities = capacities[:, None]
    demands = np.where(demands < exp_capacities, demands, exp_capacities)
    depot = depot / GRID_SIZE
    customers = customers / GRID_SIZE
    return [VRPInstance(tuple(inst_depot), list(inst_customers),
                        list(inst_demands), capacity)
            for inst_depot, inst_customers, inst_demands, capacity
            in zip(depot, customers, demands, capacities)]
