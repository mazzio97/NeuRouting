from typing import Tuple, List

import numpy as np

from nlns.instances import VRPSolution
from nlns.operators import DestroyProcedure


class TourDestroy(DestroyProcedure):
    """Tour based destroy. Remove all tours closest to a randomly selected point from a solution."""

    def __init__(self, percentage: float, point: Tuple[float, float] = None):
        assert 0 <= percentage <= 1
        self.percentage = percentage
        if point is None:
            self.point = lambda: np.random.rand(1, 2)
        else:
            self.point = lambda: point

    def __call__(self, solutions: List[VRPSolution]):
        return [ self._destroy(s) for s in solutions ]

    def _destroy(self, solution: VRPSolution):
        # Make a dictionary that maps customers to tours
        customer_to_tour = {}
        for i, tour in enumerate(solution.routes):
            for e in tour[1:-1]:
                if e in customer_to_tour:
                    customer_to_tour[e].append(i + 1)
                else:
                    customer_to_tour[e] = [i + 1]
        tours_to_remove = []
        n = solution.instance.n_customers
        n_to_remove = int(n * self.percentage)  # Number of customer that should be removed
        n_removed = 0
        customers = np.array(solution.instance.customers)
        dist = np.sum((customers - self.point()) ** 2, axis=1)
        closest_customers = np.argsort(dist) + 1
        # Iterate over customers starting with the customer closest to the random point.
        for customer_idx in closest_customers:
            # Iterate over the tours of the customer
            for i in customer_to_tour[customer_idx]:
                # and if the tour is not yet marked for removal
                if i not in tours_to_remove:
                    # mark it for removal
                    tours_to_remove.append(i)
                    n_removed += len(solution.routes[i - 1]) - 2
            # Stop once enough tours are marked for removal
            if n_removed >= n_to_remove and len(tours_to_remove) >= 1:
                break
        to_remove = set()
        for tour in tours_to_remove:
            for c in solution.routes[tour - 1][1:-1]:
                to_remove.add(c)
        solution.destroy_nodes(to_remove=list(to_remove))
