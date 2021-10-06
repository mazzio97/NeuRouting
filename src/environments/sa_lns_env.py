import math
import time
from copy import deepcopy
from typing import List

import numpy as np

from environments.lns_env import LNSEnvironment
from nlns import LNSOperator
from nlns.initial import nearest_neighbor_solution


class SimAnnealingLNSEnvironment(LNSEnvironment):
    def __init__(self,
                 operators: List[LNSOperator],
                 neighborhood_size: int,
                 initial=nearest_neighbor_solution,
                 reset_percentage: float = 0.8,
                 n_reheating: int = 5,
                 adaptive=False,
                 name="lns_sa"):
        super(SimAnnealingLNSEnvironment, self).__init__(operators, neighborhood_size, initial, adaptive, name)
        self.reset_percentage = reset_percentage
        self.n_reheating = n_reheating

    def step(self):
        reheating_time = time.time()
        reheat = True
        t_max, t_factor, temp = 0, 0, 0
        criteria = {}
        # Repeat until the time limit of one reheating iteration is reached
        while time.time() - reheating_time < self.time_limit / self.n_reheating:
            # Set a certain percentage of the data/solutions in the envs to the last accepted solution
            for i in range(int(self.reset_percentage * self.neighborhood_size)):
                self.neighborhood[i] = deepcopy(self.solution)
            criteria = super(SimAnnealingLNSEnvironment, self).step()
            # Calculate the t_max and t_factor values for simulated annealing in the first iteration
            if reheat:
                q75, q25 = np.percentile(self.neighborhood_costs, [75, 25])
                t_min = 10
                t_max = q75 - q25 + t_min
                t_factor = -math.log(t_max / t_min)
                reheat = False
            # Calculate simulated annealing temperature
            temp = t_max * math.exp(t_factor * (time.time() - reheating_time) / (self.time_limit / self.n_reheating))
        return {**criteria, "temperature": temp}

    def acceptance_criteria(self, criteria: dict) -> bool:
        current_cost = self.solution.cost()
        _, cost, temp = criteria.values()
        return cost < current_cost or np.random.rand() < math.exp(-(cost - current_cost) / temp)
