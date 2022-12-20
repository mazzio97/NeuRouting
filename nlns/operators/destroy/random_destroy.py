from typing import List
import numpy as np

from nlns.instances import VRPSolution
from nlns.operators import DestroyProcedure


class RandomDestroy(DestroyProcedure):
    """Random destroy. Select customers that should be removed at random and remove them from tours."""
    def _destroy(self, solution: VRPSolution):
        n = solution.instance.n_customers
        random_customers = np.random.choice(range(1, n + 1), int(n * self.percentage), replace=False)
        solution.destroy_nodes(to_remove=random_customers)

    def __call__(self, solutions: List[VRPSolution]):
        return [ self._destroy(s) for s in solutions ]
