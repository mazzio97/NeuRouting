from typing import List, Sequence, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import nlns
from nlns.operators import LNSOperator
from nlns.operators.neural import TorchReproducibilityMixin
from nlns.instances import VRPSolution
from nlns.models import ResidualGatedGCNModel
from nlns.utils.visualize import plot_heatmap


class HeatmapDestroy(TorchReproducibilityMixin, LNSOperator):
    """Destroy solutions by removing less likely edges.

    A heatmap is generated through a residual gated graph convolutional
    network. Edges with the lowest score are removed.
    """

    def __init__(self, percentage: float, model: ResidualGatedGCNModel,
                 num_neighbors=-1, device='cpu'):
        self.device = device
        self.model = model.to(device)
        self.num_neighbors = num_neighbors
        self.current_instances = None
        self.edges_probs = None
        self.percentage = percentage

        self.np_rng = nlns.numpy_generator_from_rng(self.rng)

    def set_random_state(self, seed: Optional[int]):
        """Set random state, enable reproducibility for torch model.

        Args:
            seed: An integer used to seed all the required generators.
                Conversely to other operators that accept a variety of
                types, it must be an integer, as torch only accepts
                integer seeds. TODO: Pass ``None`` to disable torch
                reproducibility.
        """
        super().set_random_state(seed)
        self.init_torch_reproducibility(seed)
        self.np_rng = nlns.numpy_generator_from_rng(self.rng)

    @staticmethod
    def plot_solution_heatmap(instance, sol_edges, sol_edges_probs):
        n_nodes = instance.n_customers + 1
        heatmap = np.zeros((n_nodes, n_nodes))
        pos_heatmap = np.zeros((n_nodes, n_nodes))
        for (c1, c2), prob in zip(sol_edges, sol_edges_probs):
            heatmap[c1, c2] = prob
            heatmap[c2, c1] = prob
            pos_heatmap[c1, c2] = 1 - prob
            pos_heatmap[c2, c1] = 1 - prob
        plot_heatmap(plt.gca(), instance, pos_heatmap, title='Heatmap')
        plt.show()

    def remove_edges(self, sol, sol_edges, sol_edges_probs):
        sol_edges_probs_norm = sol_edges_probs / sol_edges_probs.sum()
        n_edges = len(sol_edges)
        n_remove = int(n_edges * self.percentage)
        to_remove_idx = self.np_rng.choice(range(n_edges), size=n_remove,
                                           p=sol_edges_probs_norm,
                                           replace=False)
        sol.destroy_edges(sol_edges[to_remove_idx])

    def remove_nodes(self, sol, sol_edges, sol_edges_probs):
        sol_edges_probs_norm = sol_edges_probs / sol_edges_probs.sum()
        n_edges = len(sol_edges)
        n_nodes = sol.instance.n_customers + 1
        n_remove_nodes = int(n_nodes * self.percentage)
        to_remove_edges = self.np_rng.choice(range(n_edges), size=n_nodes,
                                             p=sol_edges_probs_norm,
                                             replace=False)
        to_remove_idx = set()
        for e in to_remove_edges:
            for c in sol_edges[e]:
                if len(to_remove_idx) == n_remove_nodes:
                    break
                if c != 0:
                    to_remove_idx.add(c)
        return sol.destroy_nodes(list(to_remove_idx))

    def call(self, solutions: Sequence[VRPSolution]) -> List[VRPSolution]:
        instances = [sol.instance for sol in solutions]
        instances = set(instances)
        if (self.current_instances is None
                or instances != self.current_instances):
            self.current_instances = instances
            instances = list(instances)
            batch_size = 64
            all_edges_preds = []
            for batch_idx in range(0, len(self.current_instances), batch_size):
                edges_preds, _ = self.model.forward(
                    *self.features(
                        instances[batch_idx:min(batch_idx + batch_size,
                                                len(self.current_instances))]))
                all_edges_preds.append(edges_preds)
            all_edges_preds = torch.cat(all_edges_preds, dim=0)
            prob_preds = torch.log_softmax(
                all_edges_preds, -1)[:, :, :, -1].to(self.device)
            self.edges_probs = np.exp(prob_preds.detach().cpu())

        for i, sol in enumerate(solutions):
            sol.verify()
            sol_edges = np.array(sol.as_edges())
            if len(self.current_instances) == 1:
                probs = self.edges_probs[0]
            else:
                probs = self.edges_probs[i]
            sol_edges_probs = np.array([1 - probs[c1, c2]
                                        for c1, c2 in sol_edges])
            # self.remove_nodes(sol, sol_edges, sol_edges_probs)
            self.remove_edges(sol, sol_edges, sol_edges_probs)

        return solutions

    def __call__(self, solutions: Sequence[VRPSolution]) -> List[VRPSolution]:
        """Apply the operator.

        Args:
            solutions: Solutions to destroy.
        Returns:
            List of destroyed solutions.
        """
        with self.sync_torch_rng_state():
            with torch.no_grad():
                return self.call(solutions)

    def load_model(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, percentage: float, ckpt_path: str,
                        **kwargs) -> 'HeatmapDestroy':
        """Instatiate the operator from a pretrained model.

        Model architecture is considered to be standard (
            :attr:`nlns.models.res_gated_gcn.default_config`).

        Keyword arguments are passed to the constructor. See
        :class:`HeatmapDestroy` or :meth:`__init__` specifications.

        Args:
            ckpt_path: Path to the model checkpoint.

        Returns:
            The instatiated operator.
        """
        operator = cls(percentage, nn.DataParallel(ResidualGatedGCNModel()),
                       **kwargs)
        operator.load_model(ckpt_path)
        return operator

    def features(self, instances):
        edges = np.stack([inst.adjacency_matrix(self.num_neighbors)
                          for inst in instances])
        edges = Variable(torch.LongTensor(edges), requires_grad=False)
        edges_values = np.stack([inst.distance_matrix for inst in instances])
        edges_values = Variable(torch.FloatTensor(edges_values),
                                requires_grad=False)
        nodes = np.zeros(instances[0].n_customers + 1, dtype=int)
        nodes[0] = 1
        nodes = Variable(torch.LongTensor(
            np.stack([nodes for _ in instances])), requires_grad=False)
        nodes_coord = np.stack([np.array([inst.depot] + inst.customers)
                                for inst in instances])
        nodes_demands = np.stack([np.array(inst.demands, dtype=float)
                                  / inst.capacity for inst in instances])
        nodes_values = np.concatenate(
            (nodes_coord, np.pad(nodes_demands, ((0, 0), (1, 0)))[:, :, None]),
            -1)
        nodes_values = Variable(torch.FloatTensor(nodes_values),
                                requires_grad=False)
        return (edges.to(self.device), edges_values.to(self.device),
                nodes.to(self.device), nodes_values.to(self.device))

    @staticmethod
    def _mean_tour_len_edges(edges_values, edges_preds):
        y = F.softmax(edges_preds, dim=-1)  # B x V x V x voc_edges
        y = y.argmax(dim=3)  # B x V x V
        # Divide by 2 because edges_values is symmetric
        tour_lens = ((y.float() * edges_values.float()).sum(dim=1).sum(dim=1)
                     / 2)
        mean_tour_len = (tour_lens.sum().to(dtype=torch.float).item()
                         / tour_lens.numel())
        return mean_tour_len
