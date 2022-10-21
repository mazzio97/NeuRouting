from copy import deepcopy
from typing import Tuple, List

import numpy as np
import torch
from torch import optim
from torch_geometric.data import DataLoader

from instances import VRPSolution
from nlns import DestroyProcedure
from nlns.neural import NeuralProcedure
from buffer import Buffer
from running_mean_std import RunningMeanStd
from egate_model import EgateModel


class EgateDestroy(NeuralProcedure, DestroyProcedure):
    def __init__(self, model: EgateModel, percentage: float, device="cpu", logger=None):
        super().__init__(model, device, logger)
        assert 0 <= percentage <= 1
        self.percentage = percentage

    def multiple(self, solutions: List[VRPSolution]):
        n_nodes = solutions[0].instance.n_customers + 1
        n_remove = int(n_nodes * self.percentage)
        nodes, edges = zip(*[self.features(sol) for sol in solutions])
        dataset = Buffer.to_dataset(nodes, edges).to(self.device)
        actions, log_p, values, entropy = self.model(dataset, n_remove, greedy=self._val_phase)
        to_remove = actions.tolist()
        for sol, remove in zip(solutions, to_remove):
            sol.destroy_nodes(remove)

    def __call__(self, solution: VRPSolution):
        self.multiple([solution])

    def _init_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.rollout_steps = 10
        self.alpha = 1.0
        self.rewards = []
        self.losses_critic = []
        self.losses_actor = []
        self.losses = []
        self.entropies = []

    def _train_step(self, opposite_procedure, train_batch):
        batch_size = len(train_batch)

        # Rollout phase
        dataset = self._rollout(train_batch, opposite_procedure, self.rollout_steps)

        # Training phase
        self.model.train()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch in data_loader:
            batch = batch.to(self.device)
            batch_size = batch.num_graphs
            actions = batch.action.reshape((batch_size, -1))
            log_p, v, entropy = self.model.evaluate(batch, actions)
            self.entropies.append(entropy.mean().item())

            target_vs = batch.v.squeeze(-1)
            old_log_p = batch.log_prob.squeeze(-1)
            adv = batch.adv.squeeze(-1)

            critic_loss = ((v - target_vs) ** 2).mean()

            ratio = torch.exp(log_p - old_log_p)
            obj = ratio * adv
            obj_clipped = ratio.clamp(1.0 - 0.2,
                                      1.0 + 0.2) * adv
            actor_loss = -torch.min(obj, obj_clipped).mean()

            loss = actor_loss + self.alpha * critic_loss

            self.losses.append(loss.item())
            self.losses_critic.append(critic_loss.item())
            self.losses_actor.append(actor_loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _train_info(self, epoch, batch_idx, log_interval) -> dict:
        critic_loss = np.mean(self.losses_critic[-log_interval:])
        actor_loss = np.mean(self.losses_actor[-log_interval:])
        loss = np.mean(self.losses[-log_interval:])
        entropy = np.mean(self.entropies[-log_interval:])
        mean_reward = np.mean(self.rewards[-log_interval:])
        return {"epoch": epoch + 1,
                "batch_idx": batch_idx + 1,
                "mean_reward": mean_reward,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "loss": loss,
                "entropy": entropy}

    def _ckpt_info(self, epoch, batch_idx) -> dict:
        return {"epoch": epoch + 1,
                "batch_idx": batch_idx + 1,
                "parameters": self.model.state_dict(),
                "optim": self.optimizer.state_dict()}

    def _rollout(self, solutions, repair_procedure, n_steps):
        n_nodes = solutions[0].instance.n_customers
        n_remove = int(n_nodes * self.percentage)
        all_nodes, all_edges = zip(*[self.features(sol) for sol in solutions])
        buffer = Buffer()
        reward_norm = RunningMeanStd()
        with torch.no_grad():
            self.model.eval()
            for i in range(n_steps):
                dataset = buffer.to_dataset(all_nodes, all_edges).to(self.device)
                actions, log_p, values, entropy = self.model(dataset, n_remove, greedy=False)

                backup_copies = [deepcopy(sol) for sol in solutions]
                prev_costs = [sol.cost() for sol in backup_copies]
                for sol, to_remove in zip(solutions, actions):
                    sol.destroy_nodes(to_remove)
                repair_procedure.multiple(solutions)

                new_all_nodes, new_all_edges, batch_rewards = [], [], []
                for j, sol in enumerate(solutions):
                    new_cost = sol.cost()
                    if new_cost < prev_costs[j]:
                        nodes, edges = self.features(sol)
                    else:
                        nodes, edges = all_nodes[j], all_edges[j]
                        solutions[j] = backup_copies[j]
                    new_all_nodes.append(nodes)
                    new_all_edges.append(edges)
                    batch_rewards.append(prev_costs[j] - new_cost)
                batch_rewards = np.array(batch_rewards)
                batch_rewards = reward_norm(batch_rewards)

                buffer.obs(all_nodes, all_edges, actions.cpu().numpy(), batch_rewards,
                           log_p.cpu().numpy(), values.cpu().numpy())
                all_nodes, all_edges = new_all_nodes, new_all_edges
        self.rewards.append(batch_rewards.mean())
        return buffer.generate_dataset()

    @staticmethod
    def features(solution: VRPSolution) -> Tuple[np.ndarray, np.ndarray]:
        inst = solution.instance
        n = inst.n_customers + 1
        nodes = np.zeros((n, 5))
        for i in range(1, n):
            sol_route = solution.get_customer_route(i)
            nodes[i] = [float(inst.demands[i - 1]) / inst.capacity,
                        float(sol_route.total_demand()) / inst.capacity,
                        float(sol_route.demand_till_customer(i)) / inst.capacity,
                        sol_route.distance_till_customer(i) / 2**0.5,
                        sol_route.total_distance() / 2**0.5]
        edges = np.zeros((n, n))
        for i, j in solution.as_edges():
            edges[i][j] = 1
        edges = np.stack([inst.distance_matrix, edges], axis=-1)
        edges = edges.reshape(-1, 2)
        return nodes, edges
