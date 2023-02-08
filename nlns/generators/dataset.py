from typing import Union, List, Tuple, Iterable
import threading
import queue
import torch
from torch_geometric.data import Data
import numpy as np
from itertools import chain, repeat
from more_itertools import chunked
import math
from uuid import uuid4
import os
from glob import glob

from nlns.generators.nazari_generator import generate_nazari_instance
from nlns.instances import VRPInstance, VRPSolution, Route
from nlns.baselines import LKHSolver
from nlns.utils.vrp_io import read_vrp, write_vrp, write_solution, read_solution


class IterableVRPDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 n_instances: int,
                 n_customer: Union[int, Tuple[int, int]],
                 solve: bool = False,
                 lkh_path: str = "executables/LKH",
                 lkh_runs: int = None,
                 save_path: Union[None, str] = None):
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
            lkh_runs (int, optional): Number of runs over LKH3 solver.
            save_path (Union[None, str], optional): Output path. If None the instances are always autogenerated.
                If a str, instances are read when posible and generated from there on. Newly generated instances
                are saved to out_path directory.
        """
        self.lkh = LKHSolver(lkh_path)
        self.lkh_runs = lkh_runs
        self.nodes = n_customer
        self.n_instances = n_instances
        self.solve = solve
        self.save_path = save_path
        
        if self.save_path is not None:
            assert os.path.exists(self.save_path), "Data path doesn't exists!"
        else:
            if os.path.exists(os.path.join(self.save_path, ".instances.txt")):
                with open(os.path.join()) as f:
                    self.saved_solutions = f.readlines()
            else:
                self.saved_solutions = list()


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

        # use saved ones before generating new ones
        if self.save_path is not None:
            saved_solutions = glob(os.path.join(self.save_path, "*.sol"))
            to_generate -= len(saved_solutions)
            for saved_solution in saved_solutions:
                fname = saved_solution.split(".sol")[0]
                nodes = int(fname.split("/")[-1].split("_")[0])
                instance = read_vrp(f"{fname}.vrp")
                tours = read_solution(f"{fname}.sol", nodes)
                yield VRPSolution(instance, [Route(t, instance) for t in tours])

        try:
            saved_solutions_file = open(os.path.join(self.save_path, ".instances.txt"), "a")

            for _ in range(to_generate):
                nodes = self._sample_nodes()
                elem = self.generate_instance(nodes)

                if self.solve:
                    elem = self.lkh.solve(elem, runs=self.lkh_runs)

                if self.save_path is not None:
                    fname = f"{nodes}_{uuid4()}"
                    if self.solve:
                        write_vrp(elem.instance, os.path.join(self.save_path, f"{fname}.vrp"))
                        write_solution(elem, os.path.join(self.save_path, f"{fname}.sol"))
                    else:
                        write_vrp(elem, os.path.join(self.save_path, f"{fname}.vrp"))
                    
                    saved_solutions_file.write(fname + "\n")

                yield elem
        finally:
            saved_solutions_file.close()

class NazariDataset(IterableVRPDataset):
    """
    Nazari dataset generator.
    """
    def generate_instance(self, nodes: int) -> VRPInstance:
        return generate_nazari_instance(nodes)