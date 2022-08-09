from typing import Union, List, Tuple
from os import path as osp
import pickle

import torch
from torch_geometric.data import Dataset, Data
import numpy as np

from generators.nazari_generator import generate_nazari_instance
from instances import VRPInstance
from baselines import LKHSolver
from utils.vrp_io import write_vrp

class VRPDataset(Dataset):
    def __init__(self, 
                 out: str,
                 n_instances: int, 
                 n_customer: Union[int, Tuple[int, int]], 
                 num_neighbors: int = -1,
                 transform = None, 
                 seed: int = 42,
                 lkh_path: str = "executables/LKH"):
        self.lkh = LKHSolver(lkh_path)
        self.out = out
        self.n_instances = n_instances
        self.nodes = n_customer
        self.seed = seed
        self.num_neighbors = num_neighbors
        super().__init__(out, transform, None, None)

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

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns:
            List[str]: File names resulting from the generation process
        """
        return [f"{i}.vrp.instance" for i in range(self.n_instances)]

    @property
    def processed_file_names(self) -> List[str]:
        """
        Returns:
            List[str]: Filenames containing serialized VRPSolutions.
        """
        return [f"{i}.data.pickle" for i in range(self.n_instances)]

    def generate_instance(self) -> VRPInstance:
        """
        Returns:
            VRPInstance: Generated VRP Instance
        """
        raise NotImplementedError

    def download(self):
        """
        Download step is the data in which files are saved to self.out generation step.
        """
        for idx in range(self.n_instances):
            instance = self.generate_instance()

            # compute solution using LKH
            solution = self.lkh.solve(instance)

            with open(osp.join(self.raw_dir, f"{idx}.vrp.instance"), "wb") as f:
                pickle.dump(solution, f)

    def process(self):
        """
        Process step converts VRPInstances into Data objects that will be 
        used by torch-geometric.
        """
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, "rb") as f:
                solution = pickle.load(f)
            
            instance = solution.instance

            # nodes position
            pos = np.stack((np.array(instance.depot), *instance.customers))

            # node features are the node position along with two extra dimension
            # one is used to signal the depot (1 for the depot, 0 for all the other)
            # the other is 0 for the depot and demand / capacity for the other nodes
            x = np.concatenate((pos, np.zeros((pos.shape[0], 1))), axis=1)
            x[0, -1] = 1
            x = np.concatenate((x, np.zeros((x.shape[0], 1))), axis=1)
            x[1:, -1] = instance.demands / instance.capacity

            # edge_index is the adjacency matrix in COO format
            adj = instance.adjacency_matrix(self.num_neighbors)
            connected = np.where(adj > 1)
            # turn adjacency matrix in COO format
            edge_index = np.stack(connected)

            # edge_attr is the feature of each edge: euclidean distance between 
            # the nodes and the node attribute value according to
            # Kool et al. (2022) Deep Policy Dynamic Programming for Vehicle Routing Problems
            distance = instance.distance_matrix[connected].reshape(-1, 1)
            edge_type = adj[connected].reshape(-1, 1)
            edge_attr = np.hstack((distance, edge_type))

            # y is the target we wish to predict to i.e. the solution provided by LKH
            y = np.stack(np.where(solution.adjacency_matrix() != 0))

            data = Data(x=x, 
                        edge_index=edge_index, 
                        edge_attr=edge_attr, 
                        pos=pos, 
                        y=y,
                        num_nodes=instance.n_customers + 1)
            
            with open(processed_path, "wb") as f:
                pickle.dump(data, f)   

    def len(self) -> int:
        """
        Returns:
            int: Number of instances in the dataset
        """
        return self.n_instances

    def get(self, idx: int) -> Data:
        """
        Return the item with the specified idx

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            Data: Data item with the specified index
        """
        with open(osp.join(self.processed_dir, f"{idx}.data.pickle"), "rb") as f:
            data = pickle.load(f)
        return data

class NazariDataset(VRPDataset):

    def generate_instance(self) -> VRPInstance:
        return generate_nazari_instance(self._sample_nodes())

    # Process and download needs to be overridden like that
    # because of the interal checks performed by the Dataset
    # class
    def process(self): super().process()
    def download(self): super().download()