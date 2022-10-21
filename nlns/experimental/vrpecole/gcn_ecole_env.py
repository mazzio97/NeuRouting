import ntpath
import os
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from ecole.core.observation import NodeBipartite
from ecole.environment import Branching
from matplotlib import pyplot as plt
from torch_geometric.data import DataLoader

from baselines import SCIPSolver
from ecole_env import EcoleEnvironment, VRPInfo
from ecole_branching_samples import generate_branching_samples
from instances import VRPInstance
from bipartite_gcn import BipartiteGCNModel
from bipartite_graph_data import GraphDataset
from nlns.initial import nearest_neighbor_solution
from utils.io import read_vrp
from utils.logging import Logger


class GCNEcoleEnvironment(EcoleEnvironment):
    def __init__(self, model: BipartiteGCNModel, device: str = "cpu", logger: Optional[Logger] = None):
        super().__init__(base_env=Branching(observation_function=NodeBipartite(),
                                            information_function=VRPInfo()),
                         name="GCN Ecole")
        self.unfixed_vars = []
        self.model = model.to(device)
        self.device = device
        self.logger = logger

    def reset(self, instance, initial=None):
        self.unfixed_vars = []
        return super(GCNEcoleEnvironment, self).reset(instance, initial)

    def step(self):
        with torch.no_grad():
            row_feats = self.obs.row_features
            edge_feats = self.obs.edge_features
            col_feats = self.obs.column_features
            logits = self.model(torch.from_numpy(row_feats.astype(np.float32)).to(self.device),
                                torch.from_numpy(edge_feats.indices.astype(np.int64)).to(self.device),
                                torch.from_numpy(edge_feats.values.astype(np.float32)).view(-1, 1).to(self.device),
                                torch.from_numpy(col_feats.astype(np.float32)).to(self.device))
            action = self.action_set[logits[self.action_set.astype(np.int64)].argmax()]
            self.unfixed_vars.append(self.scip_model.getVars()[action])
            return self.env.step(action)

    def train(self, train_instances: List[VRPInstance], val_instances: List[VRPInstance],
              n_epochs: int, batch_size: int, ckpt_path: str, rollout_steps: int = 10):
        train_files = generate_branching_samples(train_instances, rollout_steps, folder="train")
        val_files = generate_branching_samples(val_instances, rollout_steps, folder="val")
        # train_files = [str(path) for path in Path('train/').glob('instance_*.pkl')]
        # val_files = [str(path) for path in Path('val/').glob('instance_*.pkl')]
        train_data = GraphDataset(train_files, rollout_steps)
        val_data = GraphDataset(val_files, rollout_steps)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        run_name = ntpath.basename(ckpt_path)
        run_name = run_name[:run_name.rfind('.')]
        if self.logger is not None:
            self.logger.new_run(run_name=run_name)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        for epoch in range(n_epochs):
            self._process(train_loader, optimizer)
            self._process(val_loader)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"parameters": self.model.state_dict()}, ckpt_path)

    def _process(self, data_loader: DataLoader, optimizer=None):
        """
        This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
        """
        mean_loss = 0
        mean_acc = 0
        phase = "train" if optimizer is not None else "val"
        n_samples_processed = 0
        with torch.set_grad_enabled(optimizer is not None):
            for batch in data_loader:
                batch = batch.to(self.device)
                # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
                logits = self.model(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
                # Index the results by the candidates, and split and pad them
                logits = self._pad_tensor(logits[batch.candidates], batch.nb_candidates)
                # Compute the usual cross-entropy classification loss
                loss = F.cross_entropy(logits, batch.candidate_choices)

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                true_scores = self._pad_tensor(batch.candidate_scores, batch.nb_candidates)
                true_bestscore = true_scores.max(dim=-1, keepdims=True).values

                predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
                accuracy = (true_scores.gather(-1, predicted_bestindex) == true_bestscore).float().mean().item()

                mean_loss += loss.item() * batch.num_graphs
                mean_acc += accuracy * batch.num_graphs
                n_samples_processed += batch.num_graphs

        mean_loss /= n_samples_processed
        mean_acc /= n_samples_processed
        if self.logger is not None:
            self.logger.log({"mean_loss": mean_loss, "mean_accuracy": mean_acc}, phase=phase)
        return mean_loss, mean_acc

    @staticmethod
    def _pad_tensor(input_, pad_sizes, pad_value=-1e8):
        """
        This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
        """
        max_pad_size = pad_sizes.max()
        output = input_.split(pad_sizes.cpu().numpy().tolist())
        output = torch.stack([F.pad(slice_, (0, max_pad_size - slice_.size(0)), 'constant', pad_value)
                              for slice_ in output], dim=0)
        return output


if __name__ == "__main__":
    inst = read_vrp("../../../res/A-n32-k5.vrp", grid_dim=100)
    default = SCIPSolver(lns_only=True)
    # default_sol = default.solve(inst, time_limit=30)
    # inst.plot(default_sol)
    # plt.show()
    scipsolver = GCNEcoleEnvironment(BipartiteGCNModel())
    gcn_sol = scipsolver.solve(inst, initial=nearest_neighbor_solution(inst), time_limit=10)
    inst.plot(gcn_sol)
    plt.show()
