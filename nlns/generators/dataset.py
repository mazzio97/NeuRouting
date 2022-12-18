from typing import Union, List, Tuple, Iterable
import threading
import queue
import torch
from torch_geometric.data import Data
import numpy as np
from itertools import chain, repeat
from more_itertools import chunked
import math

from nlns.generators.nazari_generator import generate_nazari_instance
from nlns.instances import VRPInstance, VRPSolution
from nlns.baselines import LKHSolver


class IterableVRPDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 n_instances: int,
                 n_customer: Union[int, Tuple[int, int]],
                 solve: bool = False,
                 lkh_path: str = "executables/LKH",
                 lkh_pass: int = None,
                 lkh_runs: int = None):
        """
        Implementation of torch's Dataset which takes care of
        the generation of random instances.
        Infinitely many instances are generated, until StopIteration is raised
        by a torch.data.DataLoader.

        Args:
            n_instances (int): Instances to generate.
            n_customer (Union[int, Tuple[int, int]]): Amount of customer
                to generate. Either a constant value of a tuple representing
                the range in which customers will be sampled.
            solve (bool, optional): If True instances are solved using LKH-3. Defaults to False.
            lkh_path (str, optional): Path to LKH3 solver. Defaults to "executables/LKH".
            lkh_pass (int, optional): Number of passes over LKH3 solver.
                Defaults to None which corresponds to the number of customers.
                The less the passes the better the solution.
            lkh_runs (int, optional): Number of runs over LKH3 solver.
        """
        self.lkh = LKHSolver(lkh_path)
        self.lkh_pass = lkh_pass
        self.lkh_runs = lkh_runs
        self.nodes = n_customer
        self.n_instances = n_instances
        self.solve = solve

    def _sample_nodes(self) -> int:
        """
        Returns:
            int: Number of nodes in the generated instance. The result varies depending on
                 the initialization arguments.
        """
        if type(self.nodes) == int:
            return self.nodes
        else:
            low, high = self.nodes
            return np.random.randint(low, high)

    def generate_instance(self, nodes: int) -> VRPInstance:
        """
        Args:
            nodes (int): Number of customers in the generated instance.

        Returns:
            VRPInstance: Generated VRP Instance
        """
        raise NotImplementedError

    def __iter__(self) -> Iterable[Union[VRPInstance, VRPSolution]]:
        """
        Yields:
            Iterable[Union[VRPInstance, VRPSolution]]: Generated instance in torch geometric format.
        """
        worker_info = torch.utils.data.get_worker_info()
        to_generate = self.n_instances if worker_info is None \
                      else int(math.ceil(self.n_instances / float(worker_info.num_workers)))

        for _ in range(to_generate):
            nodes = self._sample_nodes()
            elem = self.generate_instance(nodes)

            if self.solve:
                elem = self.lkh.solve(elem, max_steps=self.lkh_pass, runs=self.lkh_runs)

            yield elem


class NazariDataset(IterableVRPDataset):
    """
    Nazari dataset generator.
    """
    def generate_instance(self, nodes: int) -> VRPInstance:
        return generate_nazari_instance(nodes)