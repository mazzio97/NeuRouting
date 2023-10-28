"""Evaluate a pair of repair-destroy operators over a set of instances.

Optionally execute multiple runs with different seeds and compute
simple statistics.
"""
import argparse
import contextlib
import json
import sys
import os.path
from typing import Iterable, List, TextIO, Union, Tuple
# from statistics import mean
from operator import attrgetter

import numpy as np
import torch
import pandas as pd

from nlns.operators.initial import nearest_neighbor_solution
from nlns.operators.destroy import PointDestroy, TourDestroy, RandomDestroy
from nlns.operators.repair import GreedyRepair
from nlns.operators.destroy.heatmap import HeatmapDestroy
from nlns.operators.repair.rl_agent import RLAgentRepair
from nlns.operators.repair.scip import SCIPRepair
from nlns.search import BaseLargeNeighborhoodSearch, LNS
from nlns.instances import VRPInstance, VRPSolution, generate_instances

DESCRIPTION = __doc__


class Namespace:
    """Custom namespace for argument parsing."""
    # Operators
    destroy: str
    repair: str
    customers: int
    destroy_percentage: float

    # Saving
    raw_path: Union[str, TextIO, None] = sys.stdout
    history_path: Union[str, TextIO, None] = sys.stdout
    statistics_path: Union[str, TextIO, None] = sys.stdout

    # Instance generation
    distribution: str = 'nazari'
    instances: int = 50
    instances_seed: int = 111

    # Runs
    runs: int = 1
    runs_seed: int = 17

    # LNS hyperparameters
    neighborhood_size: int = 256
    max_time: int = 60
    max_iterations: int = 100


def evaluate(search: BaseLargeNeighborhoodSearch,
             instances: Iterable[VRPInstance],
             **kwargs) -> Tuple[List[VRPSolution], pd.DataFrame]:
    """Run a search on the given instances.

    History for each instance is returned in a ``pandas.DataFrame``.
    """
    solutions = []

    dataframes: List[pd.DataFrame] = []

    for i, instance in enumerate(instances):
        print('Solving instance', i)
        solution = search.search(instance, **kwargs)
        solutions.append(solution)

        instance_history = search.history
        instance_history['instance'] = i
        instance_df = pd.DataFrame(instance_history)
        # Save some memory
        instance_df['cost'] = instance_df['cost'].astype(np.float32)

        dataframes.append(instance_df)
    return solutions, pd.concat(dataframes)


def rlagent_checkpoint_name(destroy: str, repair: str,
                            destroy_percentage: float,
                            customers: int) -> str:
    """Retrieve a filename for the rlagent checkpoints.

    Current standard: ``destroyname-repairname-p0X-N``

    Where ``X`` is the destruction percentage and N is the number of
    customers.
    """
    destroy_percentage
    rlagent_basename = '-'.join((destroy, repair,
                                 'p0' + str(round(destroy_percentage * 100)),
                                 str(customers)))
    return os.path.join('pretrained', rlagent_basename)


def main(namespace: Namespace):
    """Run the evaluation of a pair of operators."""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Operator factories
    destroy_operator_map = {
        'point': lambda p: PointDestroy(p),
        'tour': lambda p: TourDestroy(p),
        'random': lambda p: RandomDestroy(p),
        'heatmap': lambda p: HeatmapDestroy.from_checkpoint(
            p, 'resgatedgcn100.zip',
            device=device)
    }

    # Each rlagent pairing has its own checkpoint, select which one
    # based on which destroy operator and destroy percentage is used
    rlagent_checkpoint = rlagent_checkpoint_name(
        namespace.destroy, namespace.repair, namespace.destroy_percentage,
        namespace.customers)

    repair_operator_map = {
        'scip': lambda: SCIPRepair(),
        'greedy': lambda: GreedyRepair(),
        'rlagent': lambda: RLAgentRepair.from_checkpoint(rlagent_checkpoint,
                                                         device)
    }

    instances = tuple(generate_instances(namespace.instances,
                                         namespace.customers,
                                         distribution=namespace.distribution,
                                         seed=namespace.instances_seed))

    destroy_operator = destroy_operator_map[namespace.destroy](
        namespace.destroy_percentage)
    repair_operator = repair_operator_map[namespace.repair]()

    # Get a sequence of independent seeds from the given runs_seed
    run_seed_sequence: List[float] = np.random.SeedSequence(
        namespace.runs_seed)
    run_seeds = [
        np.random.default_rng(state)
          .random()
        for state in run_seed_sequence.spawn(namespace.runs)
    ]

    # Store costs for each run, then use them to compute whatever
    # statistics
    results: List[List[float]] = [None] * namespace.runs

    run_dataframes = []
    for run_index, run_seed in enumerate(run_seeds):
        print('new run, entropy:', run_seed)
        destroy_operator.set_random_state(run_seed)
        repair_operator.set_random_state(run_seed)

        search = LNS(destroy_operator, repair_operator,
                     nearest_neighbor_solution)

        solutions, run_dataframe = evaluate(
            search,
            instances,
            neighborhood_size=namespace.neighborhood_size,
            max_time=namespace.max_time,
            max_iterations=namespace.max_iterations)

        # Discard the solutions in favor of their costs
        results[run_index] = list(map(attrgetter('cost'), solutions))

        # Gather dataframes from each run
        run_dataframe['seed'] = run_seed
        run_dataframes.append(run_dataframe)

    # Create a matrix R x N where R is the number of runs and N the
    # number of instances. Element r, n is the cost of the solution of
    # the n-th instance as computed by the r-th run.
    # Optionally save the matrix to file.
    results_matrix = np.array(results)
    if namespace.raw_path is not None:
        np.savetxt(namespace.raw_path, results_matrix)

    # Optionally save history
    if namespace.history_path is not None:
        pd.concat(run_dataframes).to_parquet(namespace.history_path)

    # Optionally save higher level statistics as well
    means = results_matrix.mean(-1)     # Average each run
    final_mean = means.mean()           # Average between runs
    final_var = means.var()             # Variance between runs

    if namespace.statistics_path is not None:
        with contextlib.ExitStack() as stack:
            stat_file = namespace.statistics_path
            if isinstance(namespace.statistics_path, str):
                stat_file = stack.enter_context(
                    open(namespace.statistics_path, 'w'))

            json.dump({'mean': final_mean, 'final_var': final_var}, stat_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-d', '--destroy', dest='destroy', required=True)
    parser.add_argument('-r', '--repair', dest='repair', required=True)
    parser.add_argument('-c', '--customers', dest='customers', type=int,
                        required=True)
    parser.add_argument('-p', '--destroy-percentage',
                        dest='destroy_percentage',
                        type=float, required=True)

    parser.add_argument('--raw-path', dest='raw_path')
    parser.add_argument('--history-path', dest='history_path')
    parser.add_argument('--statistics-path', dest='statistics_path')

    parser.add_argument('--distribution', dest='distribution')
    parser.add_argument('--instances', dest='instances', type=int)
    parser.add_argument('--instances-seed', dest='instances_seed', type=int)

    parser.add_argument('--runs', dest='runs', type=int)
    parser.add_argument('--runs-seed', dest='runs_seed', type=int)

    parser.add_argument('--neighborhood-size', dest='neighborhood_size',
                        type=int)
    parser.add_argument('-t', '--max-time', dest='max_time', type=int)
    parser.add_argument('-i', '--max-iterations', dest='max_iterations',
                        type=int)

    main(parser.parse_args(namespace=Namespace()))
