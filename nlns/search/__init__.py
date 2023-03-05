from abc import ABC, abstractmethod
from typing import List, Callable

from time import time
from copy import deepcopy
from more_itertools import minmax
import random

import numpy as np

from nlns.instances import VRPInstance, VRPSolution
from nlns.operators import DestroyProcedure, RepairProcedure, LNSProcedure
# from nlns.utils.visualize import record_gif


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
    def __init__(self, initial_solution_fn: Callable):
        self._initial_solution_fn = initial_solution_fn
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
        return self._initial_solution_fn(instance)

    def neighborhood(self, solution: VRPSolution, size: int) -> List[VRPSolution]:
        """
        Generate a neighborhood of the current solution.
        Defaults to a list of copies of the current solution.

        Args:
            solution (VRPSolution): Solution to generate the neighborhood of.
            size (int): Size of the neighborhood.

        Returns:
            List[VRPSolution]: Solution neighborhood.
        """
        return [deepcopy(solution) for _ in range(size)]

    @abstractmethod
    def destroy(self, instances: List[VRPSolution]):
        """
        Destroy the specified solutions by the specified amount in place.

        Args:
            instances (List[VRPSolution]): Instances that will be destroyed.
            perc (float): Percentage to destroy the solution.
        """
        raise NotImplementedError

    @abstractmethod
    def repair(self, instances: List[VRPSolution]):
        """
        Repair the provided instances in place.

        Args:
            instances (List[VRPSolution]): Instances to be repaired.
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
                 max_time: float = 300) -> VRPSolution:
        """
        Search for a solution using LNS search.

        Args:
            instance (VRPInstance): Instance that will be solved.
            neighborhood_size (int, optional): Neighborhood size.
                Highly dependant on the instance to be solved for maximum efficiency. Defaults to 100.
            max_time (float, optional): Maximum time allowed for the search in seconds. Defaults to 5 minutes (300s).

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
            N = self.destroy(N)
            N = self.repair(N)
            self._iteration_durations.append(time() - iteration_start)

            # retrieve best solution in neighborhood
            N_best, _ = minmax(N, key=lambda sol: sol.cost)
            self.new_solution_found(N_best, overall_best)

            if self.is_better(N_best, current_best):
                current_best = N_best

                if current_best.cost < overall_best.cost:
                    self.better_solution_found(overall_best, N_best)
                    overall_best = current_best
                    self._history.append(overall_best)

        overall_best.time_taken = time() - initial_time
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

class LNS(BaseLargeNeighborhoodSearch):
    """
    The following class describes a LNS environment in which one procedure
    for destroying and one for repairing a solution are provided.
    """
    def __init__(self,
                 initial_solution_fn: Callable,
                 destroy_procedure: DestroyProcedure,
                 repair_procedure: RepairProcedure):
        super().__init__(initial_solution_fn)
        self.destroy_procedure = destroy_procedure
        self.repair_procedure = repair_procedure

    def destroy(self, instances: List[VRPSolution]) -> List[VRPSolution]:
        return self.destroy_procedure(instances)

    def repair(self, instances: List[VRPSolution]) -> List[VRPSolution]:
        return self.repair_procedure(instances)

class MultiOperatorLNS(BaseLargeNeighborhoodSearch):
    """
    The following class describes a LNS environment in which multiple procedures
    for destroying and repairing a solution are provided.
    Destroy and repair procedures are selected at random.
    """
    def __init__(self,
                 initial_solution_fn: Callable,
                 destroy_procedures: List[DestroyProcedure],
                 repair_procedures: List[RepairProcedure]):
        super().__init__(initial_solution_fn)
        self.destroy_procedures = destroy_procedures
        self.repair_procedures = repair_procedures

    def destroy(self, instances: List[VRPSolution]) -> List[VRPSolution]:
        destroy = random.choice(self.destroy_procedures)
        return destroy(instances)

    def repair(self, instances: List[VRPSolution]) -> List[VRPSolution]:
        repair = random.choice(self.repair_procedures)
        return repair(instances)

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
                 initial_solution_fn: Callable,
                 destroy_procedures: List[DestroyProcedure],
                 repair_procedures: List[RepairProcedure],
                 alpha: float = 0.2):
        """
        Args:
            destroy_procedures (List[DestroyProcedure]): See MultiOperatorLNS.
            repair_procedures (List[RepairProcedure]): See MultiOperatorLNS.
            alpha (float, optional): Exponential moving average alpha value. Defaults to 0.2.
        """
        super().__init__(initial_solution_fn, destroy_procedures, repair_procedures)
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

    def destroy(self, instances: List[VRPSolution]) -> List[VRPSolution]:
        destroy = self._performance_select(self.destroy_procedures, self._destroy_performance)
        self._last_destroy = destroy
        return destroy(instances)

    def repair(self, instances: List[VRPSolution]) -> List[VRPSolution]:
        repair = self._performance_select(self.repair_procedures, self._repair_performance)
        self._last_repair = repair
        return repair(instances)

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
