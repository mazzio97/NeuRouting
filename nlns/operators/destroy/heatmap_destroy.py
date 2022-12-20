from typing import List, Union, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch_geometric.data import Data, Batch

from copy import deepcopy

from nlns.instances import VRPSolution
from nlns.operators import DestroyProcedure

from nlns.models.dataloader import collate_fn
from nlns.models.res_gated_gcn import ResGatedGCN

class HeatmapDestroy(DestroyProcedure):
    def __init__(self, percentage: float, path: str):
        super().__init__(percentage)
        self.path = path
        self.heatmap_model = ResGatedGCN.load_from_checkpoint(path)

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

    def heatmap(self, instance, num_neighbors=20):
        batch = collate_fn([instance], num_neighbors=num_neighbors)
        with torch.no_grad():
            preds, _ = self.heatmap_model(batch)
        return preds[0]


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
        assert type(solutions) is list, "Destroy operator needs a list of instances to destroy."

        solution_copies = [deepcopy(s) for s in solutions]
        batch = collate_fn(solution_copies, num_neighbors=num_neighbors)
        
        with torch.no_grad():
            preds, _ = self.heatmap_model(batch)

        # destroy solution
        for solution, pred in zip(solution_copies, preds):
            solution.destroy_edges(self._edges_to_remove(solution, pred))

        return solution_copies