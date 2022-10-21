from typing import List, Union, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch_geometric.data import Data, Batch

from copy import deepcopy

from nlns.instances import VRPSolution
from nlns.operators import DestroyProcedure
#from nlns.neural import NeuralProcedure

from nlns.models.dataloader import collate_fn
from nlns.models import ResidualGatedGCNModel

class ResGatedGCNDestroy(DestroyProcedure):
    def __init__(self,
                 percentage: float,
                 num_neighbours: int = 20,
                 hidden_dim: int = 300,
                 gcn_layers: int = 30,
                 mlp_layers: int = 3):
        """
        ResGatedGCN destroy method based on the work of Kool, 2019[1].
        An heatmap of nodes that should end up in the final solution is computed.
        Nodes whose probability is lower are removed from the solution.

        [1] https://arxiv.org/pdf/1803.08475.pdf

        Args:
            percentage (float): Percentage of nodes to remove.
            num_neighbours (int): Number of neighbours used to embed the instances.
            hidden_dim (int, optional): Hetmap model hyperparameter. See models.ResidualGatedGCNModel for more informations.
            gcn_layers (int, optional): Hetmap model hyperparameter. See models.ResidualGatedGCNModel for more informations.
            mlp_layers (int, optional): Hetmap model hyperparameter. See models.ResidualGatedGCNModel for more informations.
        """
        super(pl.LightningModule, self).__init__()
        self.save_hyperparameters()
        self.num_neighbours = num_neighbours
        self.heatmap_model = ResidualGatedGCNModel(hidden_dim=hidden_dim,
                                                   mlp_layers=mlp_layers,
                                                   gcn_layers=gcn_layers)

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
        solution_edges_prob = prob[solution_edges.T].detach().cpu().numpy()
        # probabilities are computed as 1 - p with p the probability of an edge
        # being in the solution
        # we can thus sample the edges that are unlikely to be in the final solution
        # and prune them
        solution_edges_prob = 1 - solution_edges_prob
        prob_norm = solution_edges_prob / solution_edges_prob.sum()  # normalization

        # randomly sample edges to remove
        n_edges = solution_edges.shape[0]
        n_remove = int(n_edges * self.percentage)
        to_remove_idx = np.random.choice(
            range(n_edges), size=n_remove, p=prob_norm, replace=False)

        return solution_edges[to_remove_idx]

    def __call__(self, solutions: List[VRPSolution], num_neighbors: int = 20) -> List[VRPSolution]:
        """
        Destroys a solution by removing those edges that are less likely
        to end up in the final solution according to the generated heatmap.

        Args:
            solutions (List[VRPSolution]): Solution destroy.
            num_neighbours (int): Number of neighbours a batch should be created with.
                Defaults to 20.
        Returns:
            List[VRPSolution]: List of destroyed solutions.
        """
        solution_copies = [deepcopy(s) for s in solutions]
        batch = collate_fn(solution_copies)
        
        with torch.no_grad():
            prob, _ = self._heatmap_model(pyg)
            prob = prob.squeeze(0)

        # remove edges
        solution.destroy_edges(self._edges_to_remove(solution, prob))

    def training_step(self, data: Batch, batch_idx: int) -> Tuple[nn.BCELoss]:
        """
        Perform a training step.

        Args:
            data (Batch): Input data.
            batch_idx (int): Batch index.

        Returns:
            Tuple[nn.BCELoss]: Loss produced by the heatmap model.
        """
        # Class weights are computed as computed by Joshi (2019)
        weights = torch.tensor([
            self.num_neighbors**2 / 2*(self.num_neighbors**2 - 2*self.num_neighbors),
            self.num_neighbors**2 / 2*(2*self.num_neighbors)
        ], dtype=torch.float)

        _, loss = self.heatmap_model(batch, weights=weights)
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        """
        Instatiante the optimizer for the heatmap model.

        Returns:
            torch.optim.Adam: Optimizer instance.
        """
        optimizer = torch.optim.Adam(self.heatmap_model.parameters(), lr=1e-3)
        return optimizer
    
    def _evaluate(self, data: List[VRPSolution]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Perform a training step on the procedure.

        Args:
            data (List[VRPSolution]): Data on which the procedure is trained.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict]: Tuple of the form (destroyed data, loss, {})
        """
        batch = self._batch_instances(data)
        probs, loss = self._heatmap_model(batch)
        return data, loss.item(), dict()