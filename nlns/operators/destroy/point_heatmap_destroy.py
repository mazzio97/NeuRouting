from typing import Tuple, List

from copy import deepcopy

import torch
import numpy as np

from nlns.instances import VRPSolution
from nlns.operators import DestroyProcedure
from nlns.models.dataloader import collate_fn
from nlns.models.res_gated_gcn import ResGatedGCN

class HeatmapPointDestroy(DestroyProcedure):

    def __init__(self, percentage: float, path: str):
        """
        Destroy method in which the heatmap produced by the neural destroy is used
        to sample a random point in the map.
        The removed nodes will be selected based on their distance to the sample point.

        Args:
            percentage (float): Percentage of destruction.
            path (str): Path to the neural heatmap model.
        """
        super().__init__(percentage)
        self.path = path
        self.heatmap_model = ResGatedGCN.load_from_checkpoint(path)

    def _nodes_to_remove(self, solution: VRPSolution, prob: torch.Tensor) -> List:
        """
        Take the nodes close to the point around which removal will be performed.
        This is done by taking the least-likely edge to be found in the final
        solution and remove all the edges of the customers close to that point.
        The point is computed as the mid-point between the nodes at the extreme
        of the least-likely edge computed by the model.

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
        to_remove_idx = np.unravel_index(np.random.choice(range(n_edges), size=1, p=prob_norm),
                                         prob.shape)

        # the position of the edge is computed as the midpoint
        # between the nodes that are part of it
        instance_positions = np.vstack((solution.instance.depot, solution.instance.customers))
        n1_x, n1_y = tuple(instance_positions[int(to_remove_idx[0])])
        n2_x, n2_y = tuple(instance_positions[int(to_remove_idx[1])])
        point = ((n1_x + n2_x) / 2, (n1_y + n2_y) / 2)

        customers = np.array(solution.instance.customers)
        dist = np.sum((customers - point) ** 2, axis=1)
        closest_customers = np.argsort(dist)[:n_remove] + 1
        return closest_customers

    def __call__(self, solutions: List[VRPSolution], num_neighbors: int = 20) -> List[VRPSolution]:
        """
        Destroys a solution by removing those edges from the nodes that are closer to
        the less likely edge to be found in the final solution.

        Args:
            solutions (List[VRPSolution]): Solution to destroy.
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
            solution.destroy_nodes(to_remove=self._nodes_to_remove(solution, pred))

        return solution_copies


    
        
