from typing import List, Tuple, Union, Dict
from functools import partial
import os

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Data, Batch
import pytorch_lightning as pl
import numpy as np

from nlns.instances import VRPInstance, VRPSolution
from nlns.generators.dataset import NazariDataset

def instance_to_PyG(sample: VRPSolution, 
                    num_neighbors: int = 20) -> Data:
    """
    Convert a VRPInstance and optionally a VRPSolution to a torch geometric
    data instance.

    Args:
        sample (VRPSolution): VRP instance to be converted.
        num_neighbors (int, optional): Number of neighbours edges embedded for each node. Defaults to 20.

    Returns:
        Data: torch geometric data instance
    """
    instance, solution = sample

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
    
    # y is the target we wish to predict to i.e. the solution provided by LKH
    sol_adj = torch.tensor(solution.adjacency_matrix(),
                            dtype=torch.float)
    data.y = sol_adj[tuple(edge_index)]

    return data

def collate_fn(samples: List[VRPSolution], num_neighbors: int = 20) -> Batch:
  """
  Collate a list of samples in a single batch.

  Args:
      samples (List[VRPSolution]): List of samples to batch.
      num_neighbors (int, optional): Number of neighbours used to batch data. Defaults to 20.

  Returns:
      Batch: _description_
  """
  return Batch.from_data_list([
      instance_to_PyG(s, num_neighbors) for s in samples
  ])

class DataModule(pl.LightningDataModule):
  def __init__(self, 
               num_nodes: Union[int, Tuple[int, int]],
               valid_instances: int = 10_000,
               generator: str = "nazari",
               generator_params: Dict = { "solve": True, "lkh_runs": 1, "lkh_pass": 3 },
               num_neighbors: int = 20,
               batch_size: int = 32,
               steps_per_epoch: int = 500):
    super().__init__()
    self.num_nodes = num_nodes
    self.num_neighbors = num_neighbors
    self.batch_size = batch_size
    self.valid_instances = valid_instances
    self.steps_per_epoch = steps_per_epoch
    
    if generator == "nazari":
      self.generator_cls = NazariDataset
      self.generator_params = generator_params
    else:
      raise AssertionError("Distribution needs to be in ['nazari'].")

  def prepare_data(self):
    """
    Compute fixed valid and test dataset.
    """
    self.train_dataset = self.generator_cls(
      self.steps_per_epoch * self.batch_size,
      self.num_nodes, 
      **self.generator_params)

    # keep the validation set in memory
    self.valid_dataset = list(self.generator_cls(self.valid_instances, 
                                                 self.num_nodes, 
                                                 **self.generator_params))
    
  def train_dataloader(self):
    workers = os.cpu_count()

    return DataLoader(self.train_dataset, 
                      batch_size=self.batch_size, 
                      collate_fn=partial(collate_fn, num_neighbors=self.num_neighbors),
                      num_workers=workers,
                      persistent_workers=True,
                      prefetch_factor=(self.steps_per_epoch // workers))

  def val_dataloader(self):
    return DataLoader(self.valid_dataset,
                      batch_size=self.batch_size, 
                      collate_fn=partial(collate_fn, num_neighbors=self.num_neighbors),
                      num_workers=os.cpu_count())
