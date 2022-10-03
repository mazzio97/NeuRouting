from typing import Union, List, Tuple, Iterable
import torch
from torch_geometric.data import Data
import numpy as np

from generators.nazari_generator import generate_nazari_instance
from instances import VRPInstance, VRPSolution
from baselines import LKHSolver


def instance_to_PyG(instance: VRPInstance, 
                    solution: VRPSolution = None, 
                    num_neighbors: int = 20) -> Data:
    """
    Convert a VRPInstance and optionally a VRPSolution to a torch geometric
    data instance.

    Args:
        instance (VRPInstance): VRP instance to be converted.
        solution (VRPSolution, optional): VRP solution to embed in the instance data. Defaults to None.
        num_neighbors (int, optional): Number of neighbours edges embedded for each node. Defaults to 20.

    Returns:
        Data: torch geometric data instance
    """
    # nodes position
    pos = torch.tensor(np.stack((np.array(instance.depot), *instance.customers)),
                        dtype=torch.float)
    # node features are the node position along with two extra dimension
    # one is used to signal the depot (1 for the depot, 0 for all the other)
    # the other is 0 for the depot and demand / capacity for the other nodes
    x = torch.cat((pos, torch.zeros((pos.shape[0], 1), dtype=torch.float)), axis=1)
    x[0, -1] = 1
    x = torch.cat((x, torch.zeros((x.shape[0], 1), dtype=torch.float)), axis=1)
    x[1:, -1] = torch.tensor(instance.demands / instance.capacity, dtype=torch.float)
    # edge_index is the adjacency matrix in COO format
    adj = torch.tensor(instance.adjacency_matrix(num_neighbors), dtype=torch.float)
    connected = adj > 0
    # turn adjacency matrix in COO format
    edge_index = torch.stack(torch.where(connected))
    # edge_attr is the feature of each edge: euclidean distance between 
    # the nodes and the node attribute value according to
    # Kool et al. (2022) Deep Policy Dynamic Programming for Vehicle Routing Problems
    distance = torch.tensor(instance.distance_matrix[connected].reshape(-1, 1),
                            dtype=torch.float)
    edge_type = adj[connected].reshape(-1, 1)
    edge_attr = torch.hstack((distance, edge_type))
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    if solution is not None:
        # y is the target we wish to predict to i.e. the solution provided by LKH
        # added as a sparse array to save memory (adjacency matrix could end up being big but
        # will mostly be sparse)
        sol_adj = torch.tensor(solution.adjacency_matrix(),
                                dtype=torch.float)
        sol_connected = (sol_adj > 0).nonzero().t()
        data.y = torch.sparse_coo_tensor(
            sol_connected, torch.ones(sol_connected.shape[1]), sol_adj.shape, dtype=torch.float)

    return data


class IterableVRPDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 n_instances: int,
                 n_customer: Union[int, Tuple[int, int]],
                 batch_size: int = 1,
                 num_neighbors: int = -1,
                 format: str = "vrp",
                 seed: int = 42,
                 lkh_path: str = "executables/LKH",
                 lkh_pass: int = None):
        """
        Implementation of torch's Dataset which takes care of
        the generation of random instances.
        The number of nodes in an instance is kept constant between
        batches to allow parallelizing the computation while keeping
        diverse samples.

        Args:
            n_instances (int): Total number of instances to generate.
            n_customer (Union[int, Tuple[int, int]]): Amount of customer
                to generate. Either a constant value of a tuple representing
                the range in which customers will be sampled.
            batch_size (int, optional): Number of instances in one batch. Defaults to 1.
            num_neighbors (int, optional): Number of neighbours that are connected to each node. 
                Defaults to -1 which corresponds to a fully connected graph.
            format (str, optional): Format of output data. Either vrp for VRPSolution or pyg for
                torch geometric Data objects.
            seed (int, optional): Random seed. Defaults to 42.
            lkh_path (str, optional): Path to LKH3 solver. Defaults to "executables/LKH".
            lkh_pass (int, optional): Number of passes over LKH3 solver. 
                Defaults to None which corresponds to the number of customers.
                The less the passes the better the solution.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.lkh = LKHSolver(lkh_path)
        self.lkh_pass = lkh_pass
        self.instances = n_instances
        self.nodes = n_customer
        self.seed = seed
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.format = format
    
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

    def __iter__(self) -> Iterable[Data]:
        """
        Yields:
            Iterator[Iterable[Data]]: Generated instance in torch geometric format.
        """
        for _ in range(self.instances):
            nodes = self._sample_nodes()

            for _ in range(self.batch_size):
                instance = self.generate_instance(nodes)
                solution = self.lkh.solve(instance, max_steps=self.lkh_pass)

                if self.format == "pyg":
                    yield instance_to_PyG(instance, solution, num_neighbors=self.num_neighbors)
                elif self.format == "vrp":
                    yield solution

    def __len__(self) -> int:
        """
        Returns:
            int: Number of instances to be generated.
        """
        return self.instances


class NazariDataset(IterableVRPDataset):
    def generate_instance(self, nodes: int) -> VRPInstance:
        return generate_nazari_instance(nodes)