from typing import List, Union, Tuple, Dict
from utils.logging import Logger

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from sklearn.utils import compute_class_weight
from torch import optim
from torch.autograd import Variable
from torch_geometric import nn as gnn
from torch_geometric.data import Data, Batch

from generators.dataset import instance_to_PyG

from baselines import LKHSolver
from instances import VRPSolution
from nlns import DestroyProcedure
from nlns.neural import NeuralProcedure
from models import ResidualGatedGCNModel
from utils.visualize import plot_heatmap

class HeatmapModel:
    def __init__(self, 
                 device: str, 
                 hidden_dim: int = 300, 
                 gcn_layers: int = 30, 
                 mlp_layers: int = 3):
        self.device = device
        self.model = ResidualGatedGCNModel(hidden_dim=hidden_dim,
                                           mlp_layers=mlp_layers,
                                           gcn_layers=gcn_layers)
        self.model.to(device)

    def __call__(self, data: Data):
        data = data.to(self.device)
        pred, _ = self.model(data)
        return torch.sparse_coo_tensor(data.edge_index, pred)

class ResGatedGCNDestroy(NeuralProcedure, DestroyProcedure):

    def __init__(self, 
                 percentage: float, 
                 num_neighbors: int = 20,
                 model_config: dict = {"device": "cpu"}):
        super().__init__(percentage)

        self.device = model_config.pop("device", "cpu")
        self._heatmap_model = ResidualGatedGCNModel(**model_config)
        self._heatmap_model.to(self.device)

        self.num_neighbors = num_neighbors

    def _edges_to_remove(self, solution: VRPSolution, prob: torch.Tensor) -> List:
        """
        Randomly sample the edges to remove from a solution given the estimated
        probability of a node ending up in the final solution.

        Args:
            solution (VRPSolution): Proposed solution
            prob (torch.Tensor): Estimated probability

        Returns:
            List[int]: List of edges that to remove from the proposed solution
        """
        # get probability of edges in the solution
        solution_edges = np.array(solution.as_edges())
        solution_edges_prob = prob.to_dense()[solution_edges.T].detach().cpu().numpy()
        # probabilities are computed as 1 - p with p the probability of an edge
        # being in the solution
        # we can thus sample the edges that are unlikely to be in the final solution
        # and prune them
        solution_edges_prob = 1 - solution_edges_prob
        prob_norm = solution_edges_prob / solution_edges_prob.sum() # normalization
        
        # randomly sample edges to remove
        n_edges = solution_edges.shape[0]
        n_remove = int(n_edges * self.percentage)
        to_remove_idx = np.random.choice(range(n_edges), size=n_remove, p=prob_norm, replace=False)

        return solution_edges[to_remove_idx]
        
    def __call__(self, solution: VRPSolution):
        """
        Destroy a solution by removing those edges that are less likely
        to end up in the final solution according to the generated heatmap.

        Args:
            solution (VRPSolution): Current solution to be destroyed.
        """
        pyg = instance_to_PyG(solution.instance, solution, self.num_neighbors).to(self.device)
        with torch.no_grad():
            prob = self._heatmap_model(pyg)
            # TODO: Check for unbatching

        # remove edges
        solution.destroy_edges(self._edges_to_remove(solution, prob.cpu()))

    def _batch_instances(self, solutions: List[VRPSolution]) -> Batch:
        """
        Turn list of solutions into batched data.

        Args:
            solutions (List[VRPSolution]): Solutions that will be batched together

        Returns:
            Batch: Batched solutions.
        """
        batch = Batch.from_data_list([
            instance_to_PyG(s.instance, s, self.num_neighbors) for s in solutions
        ]).to(self.device)
        return batch

    def multiple(self, solutions: List[VRPSolution]):
        """
        Destroy multiple solutions.

        Args:
            solutions (List[VRPSolution]): Solutions that will be destroyed.
        """
        batch = self._batch_instances(solutions)
        with torch.no_grad():
            probs, _ = self._heatmap_model(batch)

        for idx, s in enumerate(solutions):
            s.destroy_edges(self._edges_to_remove(s, probs[idx]))

    def state_dict(self) -> Dict:
        return {
            "model_state_dict": self._heatmap_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_neighbors": self.num_neighbors,
            "percentage": self.percentage
        }
    
    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        state_dict = torch.load(path, map_location=device)
        res = cls(state_dict["percentage"], 
                  num_neighbors=state_dict["num_neighbors"],
                  model_config={"device": device})
        res._heatmap_model.load_state_dict(state_dict["model_state_dict"])
        res._init_train()
        res.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        return res

    def _init_train(self):
        self.optimizer = optim.Adam(self._heatmap_model.parameters(), lr=1e-3)
        self._heatmap_model.train()

    def _train_step(self, data: List[VRPSolution]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        assert hasattr(self, "optimizer"), "You need to call _init_train before calling _train_step."
        batch = self._batch_instances(data)
        probs, loss = self._heatmap_model(batch)

        for idx, s in enumerate(data):
            s.destroy_edges(self._edges_to_remove(s, probs[idx]))
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return data, loss, dict()

    @staticmethod
    def plot_solution_heatmap(instance, heatmap):
        plot_heatmap(plt.gca(), instance, heatmap.to_dense(), title="Heatmap")
        plt.show()

    @staticmethod
    def _mean_tour_len_edges(edges_values, edges_preds):
        y = F.softmax(edges_preds, dim=-1)  # B x V x V x voc_edges
        y = y.argmax(dim=3)  # B x V x V
        # Divide by 2 because edges_values is symmetric
        tour_lens = (y.float() * edges_values.float()).sum(dim=1).sum(dim=1) / 2
        mean_tour_len = tour_lens.sum().to(dtype=torch.float).item() / tour_lens.numel()
        return mean_tour_len
