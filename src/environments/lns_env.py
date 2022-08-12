from abc import ABC, abstractmethod
from typing import Iterable, Tuple, List

from time import time
from copy import deepcopy
from more_itertools import minmax
import random

import numpy as np
import torch

from environments import VRPEnvironment
from instances import VRPInstance, VRPSolution, VRPNeuralSolution
from nlns import DestroyProcedure, RepairProcedure, LNSOperator, LNSProcedure
from nlns.initial import nearest_neighbor_solution
from utils.visualize import record_gif


class BaseLargeNeighborhoodSearch(ABC):
    """
    Base abstract class to define a Large Neighborhood Search environment.
    The class exposes methods to perform the underlying schema in every
    LNS optimization task summarized as:
        Best = Sol = generated initial solution
        while max execution time not reached:
            N = generate a neighborhood of Sol
            N = destroy(N, destroy percentage)
            N = repair(N)
            if min(cost(N)) < cost(Sol) -> Sol = argmin(cost(N))
            if acceptance_criteria(N, Best) -> Best = argmin(cost(N))
    """
    def __init__(self):
        self._history = list()
        self._iteration_durations = list()

    def initial_solution(self, instance: VRPInstance) -> VRPSolution:
        """
        Generate the initial solution. If not subclassed defaults in using a greedy procedure
        in which each node is connected to its neighrest neighbour.

        Args:
            instance (VRPInstance): Input instance.

        Returns:
            VRPSolution: Initial solution.
        """
        return nearest_neighbor_solution(instance)

    def neighborhood(self, solution: VRPSolution, size: int) -> Iterable[VRPSolution]:
        """
        Generate a neighborhood of the current solution.
        Defaults to a list of copies of the current solution.
        
        Args:
            solution (VRPSolution): Solution to generate the neighborhood of.
            size (int): Size of the neighborhood.

        Returns:
            Iterable[VRPSolution]: Solution neighborhood.
        """
        return (deepcopy(solution) for _ in range(size))

    @abstractmethod
    def destroy(self, instance: VRPSolution, perc: float) -> VRPSolution:
        """
        Destroy the specified solution by the specified amount.

        Args:
            instance (VRPSolution): Instance that will be destroyed.
            perc (float): Percentage to destroy the solution.

        Returns:
            VRPSolution: Partially destroyed solution.
        """
        raise NotImplementedError

    @abstractmethod
    def repair(self, instance: VRPInstance) -> VRPInstance:
        """
        Repair the provided instance.

        Args:
            instance (VRPSolution): Instance to be destroyed.

        Returns:
            VRPSolution: Solution generated from repairing the provided instance.
        """
        raise NotImplementedError

    def is_better(self, x: VRPSolution, y: VRPSolution) -> bool:
        """
        Implements the acceptance criteria of a solution given a second one.
        If not overridden defaults to straight comparison between solutions
        cost.

        Args:
            x (VRPSolution): New solution for comparison.
            y (VRPSolution): Old solution.

        Returns:
            bool: True if new solution is better than the old solution.
        """
        return x.cost < y.cost

    def better_solution_found(self, old_solution: VRPSolution, new_solution: VRPSolution):
        """
        Method called when a better solution than any previous iteration is found.

        Args:
            old_solution (VRPSolution): Old solution.
            new_solution (VRPSolution): New better solution.
        """
        pass

    def new_solution_found(self, old_solution: VRPSolution, new_solution: VRPSolution):
        """
        Method called when a new solution has been found.

        Args:
            old_solution (VRPSolution): Old solution.
            new_solution (VRPSolution): New better solution.
        """
        pass

    def search(self, 
                 instance: VRPInstance, 
                 neighborhood_size: int = 100,
                 max_time: float = 300,
                 destroy_perc: float = 0.2) -> VRPSolution:
        """
        Search for a solution using LNS search.

        Args:
            instance (VRPInstance): Instance that will be solved.
            neighborhood_size (int, optional): Neighborhood size. 
                Highly dependant on the instance to be solved for maximum efficiency. Defaults to 100.
            max_time (float, optional): Maximum time allowed for the search in seconds. Defaults to 5 minutes (300s).
            destroy_perc (float, optional): Percentage of the solution (0 <= p <= 1). Defaults to 0.2.

        Returns:
            VRPSolution: Best solution found in the specified time.
        """
        self._history = []
        self._iteration_durations = []
        initial_time = time()

        current_best = self.initial_solution(instance)
        overall_best = current_best
        self._history.append(overall_best)

        while (time() - initial_time) < max_time:
            N = self.neighborhood(current_best, neighborhood_size)

            iteration_start = time()
            N = map(lambda sol: self.destroy(sol, destroy_perc), N)
            N = map(lambda sol: self.repair(sol), N)
            self._iteration_durations.append(time() - iteration_start)

            # retrieve best solution in neighborhood
            N_best, _ = minmax(N, key=lambda sol: sol.cost)
            self.new_solution_found(N_best, overall_best)
            
            if self.is_better(N_best, current_best):
                current_best = N_best

                if N_best.cost < current_best.cost:
                    self.better_solution_found(overall_best, N_best)
                    overall_best = N_best
                    self._history.append(overall_best)


        return overall_best

    @property
    def history(self) -> List[VRPSolution]:
        """
        Returns:
            List[VRPSolution]: History of solutions progress
        """
        return self._history

    @property
    def iterations_duration(self) -> List[float]:
        """
        Returns:
            List[float]: The iteration durations in seconds.
        """
        return self._iteration_durations

class MultiOperatorLNS(BaseLargeNeighborhoodSearch):
    """
    The following class describes a LNS environment in which multiple procedures
    for destroying and repairing a solution are provided.
    Destroy and repair procedures are selected at random.
    """
    def __init__(self,
                 destroy_procedures: List[DestroyProcedure],
                 repair_procedures: List[RepairProcedure]):
        super().__init__()
        self.destroy_procedures = destroy_procedures
        self.repair_procedures = repair_procedures

    def destroy(self, instance: VRPSolution, destroy_perc: float) -> VRPSolution:
        destroy = random.choice(self.destroy_procedures)
        return destroy(instance, destroy_perc)

    def repair(self, instance: VRPSolution) -> VRPSolution:
        repair = random.choice(self.repair_procedures)
        return repair(instance)

class AdaptiveLNS(MultiOperatorLNS):
    """
    The following class describes a LNS environment that differs from the one
    using multiple procedures by taking into account the effectiveness of
    each procedure. The higher the procedure yields cost effective gains the
    more likely it is to be selected.

    The effectiveness of a procedure is computed as the exponential moving
    average of
        (cost(S_new) - cost(S_best)) / time_taken(S_new)
    """
    def __init__(self,
                 destroy_procedures: List[DestroyProcedure],
                 repair_procedures: List[RepairProcedure],
                 alpha: float = 0.2):
        """
        Args:
            destroy_procedures (List[DestroyProcedure]): See MultiOperatorLNS.
            repair_procedures (List[RepairProcedure]): See MultiOperatorLNS.
            alpha (float, optional): Exponential moving average alpha value. Defaults to 0.2.
        """
        super().__init__(destroy_procedures, repair_procedures)
        self.alpha = alpha

        # last destroy and repair are used to keep track of which
        # procedures has been performed on the last iteration
        self._last_destroy = None
        self._last_repair = None
        
        # keep track of performances of each procedure
        self._destroy_performance = np.full(len(self.destroy_procedures), np.inf)
        self._repair_performance = np.full(len(self.repair_procedures), np.inf)

    def _performance_select(self, procedures: List[LNSProcedure], performance: np.array) -> LNSProcedure:
        """
        Select the best procedure based on the performances record. 
        If a procedure has never been used then select it first.

        Args:
            procedures (List[LNSProcedure]): List of procedures to pick from.
            performance (np.array): Array containing procedures performance.

        Returns:
            LNSProcedure: Best performing procedure.
        """
        if np.inf in performance:
            idx = np.where(performance == np.inf)[0][0]
        else:
            # compute probability using softmax function
            # add a small epsilon to prevent 0 probability
            exp = np.exp(performance) + 1e-20
            probs = exp / exp.sum()
            idx = np.random.choice(range(len(procedures)), p=probs, size=1)[0]
        
        return procedures[idx]

    def destroy(self, instance: VRPSolution, destroy_perc: float) -> VRPSolution:
        destroy = self._performance_select(self.destroy_procedures, self._destroy_performance)
        self._last_destroy = destroy
        return destroy(instance, destroy_perc)

    def repair(self, instance: VRPSolution) -> VRPSolution:
        repair = self._performance_select(self.repair_procedures, self._repair_performance)
        self._last_repair = repair
        return repair(instance)

    def new_solution_found(self, old_solution: VRPSolution, new_solution: VRPSolution):
        """
        Update the performance of each procedure according to the defined score.
        """
        delta = (old_solution.cost - new_solution.cost) / self.iterations_duration[-1]
        
        last_destroy_idx = self.destroy_procedures.index(self._last_destroy)
        if self._destroy_performance[last_destroy_idx] == np.inf:
            self._destroy_performance[last_destroy_idx] = delta
            self._destroy_performance[last_destroy_idx] = \
                self._destroy_performance[last_destroy_idx] * (1 - self.alpha) + delta * self.alpha

        last_repair_idx = self.repair_procedures.index(self._last_repair)
        if self._repair_performance[last_repair_idx] == np.inf:
            self._repair_performance[last_repair_idx] = delta
            self._repair_performance[last_repair_idx] = \
                self._repair_performance[last_repair_idx] * (1 - self.alpha) + delta * self.alpha


"""
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
"""