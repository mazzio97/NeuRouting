from typing import Tuple

import numpy as np

from nlns.instances import VRPSolution
from nlns.operators import DestroyProcedure


class DestroyPointBased(DestroyProcedure):
    """Point based destroy.
    Select customers that should be removed based on their distance to a random point and remove them from tours."""

    def __init__(self, percentage: float, point: Tuple[float, float] = None):
        assert 0 <= percentage <= 1
        self.percentage = percentage
        self.point = lambda: np.random.rand(1, 2) if point is None else point

    def __call__(self, solution: VRPSolution):
        n = solution.instance.n_customers
        customers = np.array(solution.instance.customers)
        dist = np.sum((customers - self.point()) ** 2, axis=1)
        closest_customers = np.argsort(dist)[:int(n * self.percentage)] + 1
        solution.destroy_nodes(to_remove=closest_customers)
