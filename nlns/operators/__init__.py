from abc import abstractmethod, ABC
from typing import List

from nlns.instances import VRPSolution


class LNSProcedure(ABC):
    @abstractmethod
    def __call__(self, solution: VRPSolution):
        pass

    def multiple(self, solutions: List[VRPSolution]):
        return [self(sol) for sol in solutions]


class DestroyProcedure(LNSProcedure):
    def __init__(self, percentage: float):
        """
        Initialize the destroy procedure with the specified dedtroy percentage.

        Args:
            percentage (float): Destroy percentage.
        """
        assert 0 <= percentage <= 1
        self.percentage = percentage

    @abstractmethod
    def __call__(self, solutions: List[VRPSolution]):
        pass


class RepairProcedure(LNSProcedure):
    @abstractmethod
    def __call__(self, solutions: VRPSolution):
        pass


class LNSOperator:
    def __init__(self, destroy_procedure: DestroyProcedure, repair_procedure: RepairProcedure):
        self.destroy = destroy_procedure
        self.repair = repair_procedure
