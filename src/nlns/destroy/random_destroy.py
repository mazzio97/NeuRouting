import numpy as np

from instances import VRPSolution
from nlns import DestroyProcedure


class RandomDestroy(DestroyProcedure):
    """Random destroy. Select customers that should be removed at random and remove them from tours."""
    def __call__(self, solution: VRPSolution, percentage: float):
        assert 0 <= percentage <= 1, "Percentage must be in range [0, 1]."
        n = solution.instance.n_customers
        random_customers = np.random.choice(range(1, n + 1), int(n * percentage), replace=False)
        solution.destroy_nodes(to_remove=random_customers)
        return solution
