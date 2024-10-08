import contextlib
from typing import Sequence, Optional

import torch
import numpy as np

import torch.nn.functional as F
from torch import optim

from nlns.operators import LNSOperator
from nlns.operators.neural import TorchReproducibilityMixin, Trainable
from nlns.instances import VRPSolution
from nlns.models import VRPActorModel, VRPCriticModel, RLAgentSolution


@contextlib.contextmanager
def as_rlagent(solutions):
    neural_solutions = [RLAgentSolution.from_solution(solution)
                        for solution in solutions]

    try:
        yield neural_solutions
    finally:
        for solution, neural in zip(solutions, neural_solutions):
            solution.routes = neural.routes


class RLAgentRepair(TorchReproducibilityMixin, Trainable, LNSOperator):
    """Repair solutions by applying a deep reinforcement learning model.

    The model is an actor critic attention based approach, as defined
    by `Huttong & Tierney (2020) <https://doi.org/10.3233/FAIA200124>`_.

    Pytorch is required.
    """

    def __init__(self, actor: VRPActorModel, critic: VRPCriticModel = None,
                 device='cpu', logger=None):
        self.model = actor.to(device)
        self.critic = critic.to(device) if critic is not None else critic
        self.device = device
        self.logger = logger

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

    def call(self, solutions: Sequence[RLAgentSolution]):
        """Completely repair the given solutions.

        This method shall be used during training. For faster inference,
        use :meth:`__call__`.
        """
        emb_size = max([solution.min_nn_repr_size() for solution in solutions])
        batch_size = len(solutions)

        # Create envs input
        static_input = np.zeros((batch_size, emb_size, 2))
        dynamic_input = np.zeros((batch_size, emb_size, 2), dtype='int')
        for i, solution in enumerate(solutions):
            static_nn_input, dynamic_nn_input = (
                solution.network_representation(emb_size))
            static_input[i] = static_nn_input
            dynamic_input[i] = dynamic_nn_input

        static_input = torch.from_numpy(static_input).to(self.device).float()
        dynamic_input = torch.from_numpy(dynamic_input).to(self.device).long()
        capacity = np.fromiter(
            (solution.instance.capacity for solution in solutions),
            dtype=float)[:, np.newaxis]

        cost_estimate = None
        if self.critic is not None:
            cost_estimate = self._critic_model_forward(
                static_input, dynamic_input, capacity)

        tour_idx, tour_logp = self._actor_model_forward(
            solutions, static_input, dynamic_input, capacity)
        return tour_idx, tour_logp, cost_estimate

    def __call__(self, solutions: Sequence[VRPSolution]
                 ) -> Sequence[VRPSolution]:
        """Apply the operator to all given solutions.

        Args:
            solutions: The destroyed solutions to be repaired.

        Returns:
            The completely repaired solutions.
        """
        neural_solutions = [RLAgentSolution.from_solution(solution)
                            for solution in solutions]
        with self.sync_torch_rng_state():
            with torch.no_grad():
                self.call(neural_solutions)

        for solution, neural in zip(solutions, neural_solutions):
            solution.routes = neural.routes

        return solutions

    def init_train(self):
        self.actor_optim = optim.Adam(self.model.parameters(), lr=1e-4)
        self.model.train()
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=5e-4)
        self.critic.train()
        self.losses_actor = []
        self.losses_critic = []
        self.rewards = []
        self.diversity_values = []

    def training_step(self, train_batch):
        costs_destroyed = [solution.cost for solution in train_batch]
        with as_rlagent(train_batch) as rlagent_batch:
            _, tour_logp, critic_est = self.call(rlagent_batch)
        costs_repaired = [solution.cost for solution in train_batch]

        # Reward/Advantage computation
        reward = np.array(costs_repaired) - np.array(costs_destroyed)
        reward = torch.from_numpy(reward).float().to(self.device)
        advantage = reward - critic_est

        # Actor loss computation and backpropagation
        max_grad_norm = 2.
        actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.actor_optim.step()

        # Critic loss computation and backpropagation
        critic_loss = torch.mean(advantage ** 2)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic_optim.step()

        self.rewards.append(torch.mean(reward.detach()).item())
        self.losses_actor.append(torch.mean(actor_loss.detach()).item())
        self.losses_critic.append(torch.mean(critic_loss.detach()).item())

    def training_info(self, epoch, batch_idx, log_interval):
        mean_loss = np.mean(self.losses_actor[-log_interval:])
        mean_critic_loss = np.mean(self.losses_critic[-log_interval:])
        mean_reward = np.mean(self.rewards[-log_interval:])
        return {"epoch": epoch + 1,
                "batch_idx": batch_idx + 1,
                "mean_reward": mean_reward,
                "actor_loss": mean_loss,
                "critic_loss": mean_critic_loss}

    def checkpoint(self, epoch, batch_idx) -> dict:
        return {"epoch": epoch + 1,
                "batch_idx": batch_idx + 1,
                "parameters": self.model.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_optim": self.critic_optim.state_dict()}

    def save(self, path: str, epoch, batch_idx):
        torch.save(self.checkpoint(epoch, batch_idx), path)

    @classmethod
    def from_checkpoint(cls, path: str, device: str) -> 'RLAgentRepair':
        """Load operator from model checkpoint.

        Obtained through :meth:`save`.
        """
        checkpoint = torch.load(path, map_location=device)
        model = VRPActorModel(device=device)
        model.load_state_dict(checkpoint['parameters'])
        critic = VRPCriticModel()
        critic.load_state_dict(checkpoint['critic'])

        operator = cls(model, critic, device=device)
        operator.init_train()
        operator.actor_optim.load_state_dict(checkpoint['actor_optim'])
        operator.critic_optim.load_state_dict(checkpoint['critic_optim'])

        model.eval()

        return operator

    def _actor_model_forward(self, incomplete_solutions, static_input,
                             dynamic_input, vehicle_capacity):
        batch_size = static_input.shape[0]
        tour_idx, tour_logp = [], []

        solutions_repaired = np.zeros(batch_size)

        origin_idx = np.zeros(batch_size, dtype=int)

        while not solutions_repaired.all():
            # if origin_idx == 0 select the next tour-end that serves as the
            # origin at random
            for i, solution in enumerate(incomplete_solutions):
                if origin_idx[i] == 0 and not solutions_repaired[i]:
                    origin_idx[i] = self.rng.choice(
                        solution.incomplete_nn_idx)

            mask = self._get_mask(origin_idx, dynamic_input,
                                  incomplete_solutions, vehicle_capacity).to(
                                    self.device)
            mask = mask.float()

            # Rescale customer demand based on vehicle capacity
            dynamic_input_float = dynamic_input.float()
            dynamic_input_float[:, :, 0] = (dynamic_input_float[:, :, 0]
                                            / vehicle_capacity)

            origin_static_input = static_input[torch.arange(batch_size),
                                               origin_idx]
            origin_dynamic_input = dynamic_input_float[
                torch.arange(batch_size), origin_idx]

            # Forward pass:
            # Returns a probability distribution over the point
            # (tour end or depot) that origin should be connected to.
            probs = self.model.forward(static_input, dynamic_input_float,
                                       origin_static_input,
                                       origin_dynamic_input)
            # Set prob of masked tour ends to zero
            probs = F.softmax(probs + mask.log(), dim=1)

            if self.model.training:
                m = torch.distributions.Categorical(probs)

                ptr = m.sample()
                while not torch.gather(
                        mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy selection
                logp = prob.log()

            # Perform action for all data sequentially
            nn_input_updates = []
            ptr_np = ptr.cpu().numpy()
            for i, solution in enumerate(incomplete_solutions):
                idx_from = origin_idx[i].item()
                idx_to = ptr_np[i]
                # No need to update in this case
                if idx_from == 0 and idx_to == 0:
                    continue

                # Connect origin to select point
                nn_input_update, cur_nn_input_idx = solution.connect(idx_from,
                                                                     idx_to)

                for s in nn_input_update:
                    s.insert(0, i)
                    nn_input_updates.append(s)

                # Update origin
                if len(solution.incomplete_nn_idx) == 0:
                    solutions_repaired[i] = 1
                    # If instance is repaired set origin to 0
                    origin_idx[i] = 0
                else:
                    # Otherwise, set to tour end of the connect tour
                    origin_idx[i] = cur_nn_input_idx

            # Update network input
            nn_input_update = np.array(nn_input_updates)
            nn_input_update = torch.from_numpy(nn_input_update).to(
                self.device).long()
            dynamic_input[nn_input_update[:, 0], nn_input_update[:, 1]] = \
                nn_input_update[:, 2:]

            logp = logp * (1. - torch.from_numpy(solutions_repaired)
                                     .float().to(self.device))
            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))

        tour_idx = torch.cat(tour_idx, dim=1)
        tour_logp = torch.cat(tour_logp, dim=1)
        return tour_idx, tour_logp

    def _critic_model_forward(self, static_input, dynamic_input,
                              vehicle_capacity: int):
        dynamic_input_float = dynamic_input.float()

        dynamic_input_float[:, :, 0] = (dynamic_input_float[:, :, 0]
                                        / vehicle_capacity)

        return self.critic.forward(static_input, dynamic_input_float).view(-1)

    @staticmethod
    def _get_mask(origin_nn_input_idx, dynamic_input, solutions,
                  capacity: int):
        """Returns a mask for the current nn_input"""
        batch_size = origin_nn_input_idx.shape[0]

        # Start with all used input positions
        mask = (dynamic_input[:, :, 1] != 0).cpu().long().numpy()

        for i in range(batch_size):
            idx_from = origin_nn_input_idx[i]
            origin_tour = solutions[i].map_network_idx_to_route[idx_from][0]
            origin_pos = solutions[i].map_network_idx_to_route[idx_from][1]

            # Find the start of the tour in the nn input
            # e.g. for the tour [2, 3] two entries in nn input exists
            if origin_pos == 0:
                idx_same_tour = origin_tour[-1][2]
            else:
                idx_same_tour = origin_tour[0][2]

            mask[i, idx_same_tour] = 0

            # Do not allow origin location = destination location
            mask[i, idx_from] = 0

        mask = torch.from_numpy(mask)

        origin_demands = dynamic_input[torch.arange(batch_size),
                                       origin_nn_input_idx, 0].unsqueeze(1)
        combined_demands = (origin_demands.expand(batch_size,
                                                  dynamic_input.shape[1])
                            + dynamic_input[:, :, 0])
        mask[combined_demands.numpy() > capacity] = 0

        mask[:, 0] = 1  # Always allow to go to the depot

        return mask
