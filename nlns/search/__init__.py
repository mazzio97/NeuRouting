from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Union, Sequence

from time import time
from copy import deepcopy
from more_itertools import minmax
import random

import numpy as np

from nlns.instances import VRPInstance, VRPSolution
from nlns.operators import LNSOperator
# from nlns.utils.visualize import record_gif


class BaseLargeNeighborhoodSearch(ABC):
    """Base class to define a Large Neighborhood Search environment.

    The class exposes methods to perform the underlying schema in every
    LNS optimization task summarized as
    .. code-block:: text

        Best = Sol = generated initial solution
        while max execution time not reached:
            N = generate a neighborhood of Sol
            N = destroy(N, destroy percentage)
            N = repair(N)
            if min(cost(N)) < cost(Sol) -> Sol = argmin(cost(N))
            if acceptance_criteria(N, Best) -> Best = argmin(cost(N))
    """
    def __init__(self,
                 initial_solution_fn: Optional[Callable[[VRPInstance],
                                                        VRPSolution]] = None):
        self._initial_solution_fn = initial_solution_fn
        self._history = list()
        self._iteration_durations = list()

    def initial_solution(self, instance_or_solution:
                         Union[VRPInstance, VRPSolution]) -> VRPSolution:
        """Generate the initial solution.

        Default behaviour is to use the initial solution function
        provided at construction time.

        If a solution is provided instead of an unsolved instance,
        the solution itself is returned as it is.

        Args:
            instance: Input instance or pre-initialized solution.

        Returns:
            VRPSolution: Initial solution.

        Raises:
            ValueError: If input is an unsolved instance and no initial
                solution function was provided (``None``).
        """
        is_solution = isinstance(instance_or_solution, VRPSolution)

        if is_solution:
            return instance_or_solution

        if self._initial_solution_fn is None and not is_solution:
            raise ValueError('When input is a VRPInstance, provide an initial '
                             'solution strategy at construction time.')

        return self._initial_solution_fn(instance_or_solution)

    def neighborhood(self, solution: VRPSolution,
                     size: int) -> List[VRPSolution]:
        """Generate a neighborhood of the current solution.

        Defaults to a list of copies of the current solution.

        Args:
            solution: Solution to generate the neighborhood of.
            size: Size of the neighborhood.

        Returns:
            Solution neighborhood.
        """
        return [deepcopy(solution) for _ in range(size)]

    @abstractmethod
    def destroy(self, instances: List[VRPSolution]):
        """Destroy the specified solutions, in place.

        Args:
            instances: Instances that will be destroyed.
            perc: Percentage to destroy the solution.
        """
        raise NotImplementedError

    @abstractmethod
    def repair(self, instances: List[VRPSolution]):
        """Repair the provided instances in place.

        Args:
            instances: Instances to be repaired.
        """
        raise NotImplementedError

    def is_better(self, x: VRPSolution, y: VRPSolution) -> bool:
        """Implements the acceptance criteria of a solution
        .
        If not overridden defaults to straight comparison between
        solutions cost.

        Args:
            x: New solution for comparison.
            y: Old solution.

        Returns:
            True if new solution is better than the old solution.
        """
        return x.cost < y.cost

    def better_solution_found(self, old_solution: VRPSolution,
                              new_solution: VRPSolution):
        """Called when a better solution than any previous iteration is found.

        Args:
            old_solution: Old solution.
            new_solution: New better solution.
        """
        pass

    def new_solution_found(self, old_solution: VRPSolution,
                           new_solution: VRPSolution):
        """Method called when a new solution has been found.

        Args:
            old_solution: Old solution.
            new_solution: New better solution.
        """
        pass

    def search(self,
               instance_or_solution: Union[VRPInstance, VRPSolution],
               neighborhood_size: int = 100,
               max_time: float = 300) -> VRPSolution:
        """Search for a solution using LNS search.

        Args:
            instance: Instance that will be solved.
            neighborhood_size: Neighborhood size. Highly dependant on
                the instance to be solved for maximum efficiency.
                Defaults to 100.
            max_time: Maximum time allowed for the search in seconds.
                Defaults to 5 minutes (300s).

        Returns:
            VRPSolution: Best solution found in the specified time.
        """
        self._history = []
        self._iteration_durations = []
        initial_time = time()

        current_best = self.initial_solution(instance_or_solution)
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
        """Retrieve history of the search.

        Returns:
            List[VRPSolution]: History of solutions progress
        """
        return self._history

    @property
    def iterations_duration(self) -> List[float]:
        """Retrieve list of durations for each step.

        Returns:
            List[float]: The iteration durations in seconds.
        """
        return self._iteration_durations


class LNS(BaseLargeNeighborhoodSearch):
    """Simple search using one destroy procedure and one repair."""

    def __init__(self, destroy_procedure: LNSOperator,
                 repair_procedure: LNSOperator,
                 initial_solution_fn:
                 Optional[Callable[[VRPInstance], VRPSolution]] = None):
        super().__init__(initial_solution_fn)
        self.destroy_procedure = destroy_procedure
        self.repair_procedure = repair_procedure

    def destroy(self, instances: List[VRPSolution]) -> List[VRPSolution]:
        """Apply the given destroy operator."""
        return self.destroy_procedure(instances)

    def repair(self, instances: List[VRPSolution]) -> List[VRPSolution]:
        """Apply the given repair operator."""
        return self.repair_procedure(instances)


class MultiOperatorLNS(BaseLargeNeighborhoodSearch):
    """Search with multiple destroy and repair operators.

    The used operator is randomly chosen at each step.
    """

    def __init__(self, destroy_procedures: Sequence[LNSOperator],
                 repair_procedures: Sequence[LNSOperator],
                 initial_solution_fn:
                 Optional[Callable[[VRPInstance], VRPSolution]] = None):
        super().__init__(initial_solution_fn)
        self.destroy_procedures = destroy_procedures
        self.repair_procedures = repair_procedures

    def destroy(self, instances: List[VRPSolution]) -> List[VRPSolution]:
        """Destroy solutions with a random destroy operator."""
        destroy = random.choice(self.destroy_procedures)
        return destroy(instances)

    def repair(self, instances: List[VRPSolution]) -> List[VRPSolution]:
        """Repair solutions with a random repair operator."""
        repair = random.choice(self.repair_procedures)
        return repair(instances)


class AdaptiveLNS(MultiOperatorLNS):
    """Adaptively choose between multiple destroy and repair operators.

    Operators that prove to be more effective (provide more cost
    reduction in less time) are chosen with a higher probability.

    The effectiveness of a procedure is computed as the exponential
    moving average of
    ``(cost(S_new) - cost(S_best)) / time_taken(S_new)``.

    Args:
            destroy_procedures: See MultiOperatorLNS.
            repair_procedures: See MultiOperatorLNS.
            alpha: Exponential moving average alpha value.
                Defaults to 0.2.
    """

    def __init__(self, destroy_procedures: List[LNSOperator],
                 repair_procedures: List[LNSOperator],
                 initial_solution_fn:
                 Optional[Callable[[VRPInstance], VRPSolution]] = None,
                 alpha: float = 0.2):
        super().__init__(destroy_procedures, repair_procedures,
                         initial_solution_fn)
        self.alpha = alpha

        # last destroy and repair are used to keep track of which
        # procedures has been performed on the last iteration
        self._last_destroy = None
        self._last_repair = None

        # keep track of performances of each procedure
        self._destroy_performance = np.full(len(self.destroy_procedures),
                                            np.inf)
        self._repair_performance = np.full(len(self.repair_procedures),
                                           np.inf)

    def _performance_select(self, procedures: List[LNSOperator],
                            performance: np.array) -> LNSOperator:
        """Select the best procedure based on the performances record.

        If a procedure has never been used then select it first.

        Args:
            procedures: List of procedures to pick from.
            performance: Array containing procedures performance.

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
        """Destroy solutions with "best" destroy operator."""
        destroy = self._performance_select(self.destroy_procedures,
                                           self._destroy_performance)
        self._last_destroy = destroy
        return destroy(instances)

    def repair(self, instances: List[VRPSolution]) -> List[VRPSolution]:
        """Repair solutions with "best" repair operator."""
        repair = self._performance_select(self.repair_procedures,
                                          self._repair_performance)
        self._last_repair = repair
        return repair(instances)

    def new_solution_found(self, old_solution: VRPSolution,
                           new_solution: VRPSolution):
        """Update the performance of each operator."""
        delta = ((old_solution.cost - new_solution.cost)
                 / self.iterations_duration[-1])

        last_destroy_idx = self.destroy_procedures.index(self._last_destroy)
        if np.isinf(self._destroy_performance[last_destroy_idx]):
            self._destroy_performance[last_destroy_idx] = delta
            self._destroy_performance[last_destroy_idx] = (
                self._destroy_performance[last_destroy_idx] * (1 - self.alpha)
                + delta * self.alpha)

        last_repair_idx = self.repair_procedures.index(self._last_repair)
        if np.isinf(self._repair_performance[last_repair_idx]):
            self._repair_performance[last_repair_idx] = delta
            self._repair_performance[last_repair_idx] = (
                self._repair_performance[last_repair_idx] * (1 - self.alpha)
                + delta * self.alpha)
