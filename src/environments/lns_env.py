import time
from copy import deepcopy

import numpy as np
import torch
from typing import List, Tuple

from environments import VRPEnvironment
from instances import VRPInstance, VRPSolution, VRPNeuralSolution
from nlns import DestroyProcedure, RepairProcedure, LNSOperator
from nlns.initial import nearest_neighbor_solution
from utils.visualize import record_gif

EMA_ALPHA = 0.2  # Exponential Moving Average Alpha


class LargeNeighborhoodSearch:
    def __init__(self,
                 operators: List[LNSOperator],
                 initial=nearest_neighbor_solution,
                 adaptive=False):
        self.initial = initial
        self.operators = operators
        self.n_operators = len(operators)
        self.adaptive = adaptive
        self.performances = np.array([np.inf] * self.n_operators) if adaptive else None
        self.usages = np.zeros(self.n_operators, dtype=int) if adaptive else None

    def select_operator_pairs(self, size=1):
        if self.adaptive and self.n_operators > 1:
            if np.inf in self.performances:
                indices = np.where(self.performances == np.inf)[0]
                idxs = np.random.choice(indices, size=size)
            else:
                perf = self.performances + abs(min(self.performances))
                probs = perf / perf.sum()
                idxs = np.random.choice(range(self.n_operators), p=probs, size=size)
        else:
            idxs = np.random.randint(0, self.n_operators, size=size)

        op_sols_dict = {op_idx: np.where(idxs == op_idx)[0] for op_idx in np.unique(idxs)}
        if self.adaptive:
            for op_idx, sols_idxs in op_sols_dict.items():
                self.usages[op_idx] += len(sols_idxs)
        return op_sols_dict


class LNSEnvironment(LargeNeighborhoodSearch, VRPEnvironment):
    def __init__(self,
                 operators: List[LNSOperator],
                 neighborhood_size: int,
                 initial=nearest_neighbor_solution,
                 adaptive=True,
                 name="lns"):
        LargeNeighborhoodSearch.__init__(self, operators, initial, adaptive)
        VRPEnvironment.__init__(self, name)
        self.neighborhood_size = neighborhood_size
        self.incumbent_solution = None
        self.neighborhood = None
        self.neighborhood_costs = None
        self.history = None

    def reset(self, instance: VRPInstance, **args):
        super(LNSEnvironment, self).reset(instance)
        self.solution = self.initial(instance)
        if any([callable(getattr(op.repair, "_actor_model_forward", None)) for op in self.operators]):
            self.solution = VRPNeuralSolution.from_solution(self.solution)
        self.incumbent_solution = self.solution

    def step(self) -> dict:
        current_cost = self.solution.cost()

        op_sols_dict = self.select_operator_pairs(size=self.neighborhood_size)

        iter_start_time = time.time()
        with torch.no_grad():
            for op_idx, sols_idx in op_sols_dict.items():
                self.operators[op_idx].destroy.multiple(self.neighborhood[sols_idx])
                self.operators[op_idx].repair.multiple(self.neighborhood[sols_idx])
        lns_iter_duration = time.time() - iter_start_time

        self.neighborhood_costs = [sol.cost() for sol in self.neighborhood]
        best_idx = int(np.argmin(self.neighborhood_costs))
        for op_idx, sols_idx in op_sols_dict.items():
            if best_idx in sols_idx:
                best_op_idx = op_idx
                break
        new_cost = self.neighborhood_costs[best_idx]

        # If adaptive search is used, update performance scores
        if self.adaptive:
            delta = (current_cost - new_cost) / lns_iter_duration
            if self.performances[best_op_idx] == np.inf:
                self.performances[best_op_idx] = delta
            self.performances[best_op_idx] = self.performances[best_op_idx] * (1 - EMA_ALPHA) + delta * EMA_ALPHA

        self.n_steps += 1

        return {"best_idx": best_idx, "cost": new_cost}

    def solve(self, instance: VRPInstance, max_steps=None, time_limit=None, record=False) -> VRPSolution:
        self.reset(instance)
        initial_cost = self.solution.cost()
        self.max_steps = max_steps if max_steps is not None else self.max_steps
        self.time_limit = time_limit if time_limit is not None else self.time_limit
        self.history = [] if record else None

        start_time = time.time()
        while self.n_steps < self.max_steps and time.time() - start_time < self.time_limit:
            # Create neighborhood_size copies of the same solution that can be repaired in parallel
            self.neighborhood = np.array([deepcopy(self.solution) for _ in range(self.neighborhood_size)])
            criteria = self.step()

            if criteria["cost"] < self.incumbent_solution.cost():
                if record:
                    self.history.append(deepcopy(self.incumbent_solution))
                self.incumbent_solution = deepcopy(self.neighborhood[criteria["best_idx"]])
                self.incumbent_solution.verify()
                self.improvements += 1

            if self.acceptance_criteria(criteria):
                self.solution = deepcopy(self.neighborhood[criteria["best_idx"]])
                # self.solution.verify()

        self.solution = self.incumbent_solution
        final_cost = self.solution.cost()
        self.gap = (initial_cost - final_cost) / final_cost * 100

        if record:
            self.history.append(deepcopy(self.incumbent_solution))
            record_gif(self.history,
                       file_name=f"{self.name.lower().replace(' ', '_')}_n{self.instance.n_customers}.gif")

        if self.adaptive:
            self.usages = self.usages.astype(np.float32) / self.usages.sum()

        return self.solution

    def acceptance_criteria(self, criteria: dict) -> bool:
        # Accept a solution if the acceptance criteria is fulfilled
        return criteria["cost"] < self.solution.cost()

    def __deepcopy__(self, memo):
        return LNSEnvironment(self.operators, self.neighborhood_size, self.initial, self.adaptive, self.name)
