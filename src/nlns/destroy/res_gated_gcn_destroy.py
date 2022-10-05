from typing import List, Union, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch_geometric.data import Data, Batch

from generators.dataset import instance_to_PyG

from instances import VRPSolution
from nlns import DestroyProcedure
from nlns.neural import NeuralProcedure
from models import ResidualGatedGCNModel

class ResGatedGCNDestroy(NeuralProcedure, DestroyProcedure):

    def __init__(self,
                 percentage: float,
                 num_neighbors: int = 20,
                 model_config: Dict = {
                     "device": "cpu",
                     "hidden_dim": 300,
                     "gcn_layers": 30,
                     "mlp_layers": 3
                 }):
        """
        ResGatedGCN destroy method based on the work of Kool, 2019[1].
        An heatmap of nodes that should end up in the final solution is computed.
        Nodes whose probability is lower are removed from the solution.

        [1] https://arxiv.org/pdf/1803.08475.pdf

        Args:
            percentage (float): Percentage of nodes to remove.
            num_neighbors (int, optional): Number of neighbours edges embedded in each node. Defaults to 20.
            model_config (Dict, optional): ResGatedGCN parameters. 
                Defaults to { "device": "cpu", "hidden_dim": 300, "gcn_layers": 30, "mlp_layers": 3 }.
                See models.ResidualGatedGCNModel for more informations.
        """
        super().__init__(percentage)

        self.device = model_config.pop("device", "cpu")

        hidden_dim = model_config.pop("hidden_dim", 300)
        gcn_layers = model_config.pop("gcn_layers", 30)
        mlp_layers = model_config.pop("mlp_layers", 3)
        self._heatmap_model = ResidualGatedGCNModel(hidden_dim=hidden_dim,
                                                    mlp_layers=mlp_layers,
                                                    gcn_layers=gcn_layers)
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

    def __call__(self, solution: VRPSolution):
        """
        Destroy a solution by removing those edges that are less likely
        to end up in the final solution according to the generated heatmap.

        Args:
            solution (VRPSolution): Current solution to be destroyed.
        """
        pyg = instance_to_PyG(solution.instance, 
                              solution,
                              self.num_neighbors).to(self.device)
        
        with torch.no_grad():
            prob, _ = self._heatmap_model(pyg)
            prob = prob.squeeze(0)

        # remove edges
        solution.destroy_edges(self._edges_to_remove(solution, prob))

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
            probs, loss = self._heatmap_model(batch)

        for idx, s in enumerate(solutions):
            s.destroy_edges(self._edges_to_remove(s, probs[idx]))

    def state_dict(self) -> Dict:
        """
        Returns:
            Dict: Neural procedure state dict to save the procedure to disk for later usage.
        """
        return {
            "model_state_dict": self._heatmap_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_neighbors": self.num_neighbors,
            "percentage": self.percentage
        }

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        """
        Load the procedure from file.

        Args:
            path (str): Path of the saved procedure
            device (str, optional): Device on which he procedure is loaded. Defaults to "cpu".

        Returns:
            ResGatedGCNDestroy: The procedure instance
        """
        state_dict = torch.load(path, map_location=device)
        res = cls(state_dict["percentage"],
                  num_neighbors=state_dict["num_neighbors"],
                  model_config={"device": device})
        res._heatmap_model.load_state_dict(state_dict["model_state_dict"])
        res._init_train()
        res.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        return res

    def _init_train(self):
        """
        Initialize training of this neual procedure.
        """
        self.optimizer = optim.Adam(self._heatmap_model.parameters(), lr=1e-3)
        self._heatmap_model.train()

    def _train_step(self, data: List[VRPSolution]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Perform a training step on the procedure.

        Args:
            data (List[VRPSolution]): Data on which the procedure is trained.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict]: Tuple of the form (destroyed data, loss, {})
        """
        assert hasattr(self, "optimizer"), "Call _init_train before _train_step."
        batch = self._batch_instances(data)

        weights = torch.tensor([
            self.num_neighbors**2 / 2*(self.num_neighbors**2 - 2*self.num_neighbors),
            self.num_neighbors**2 / 2*(2*self.num_neighbors)
        ], dtype=torch.float)
        probs, loss = self._heatmap_model(batch, weights=weights)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return data, loss.item(), dict()
    
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