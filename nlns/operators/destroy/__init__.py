from typing import Sequence, Tuple, Optional, List

import numpy as np

import nlns
from nlns.instances import VRPSolution
from nlns.operators import LNSOperator

# from .heatmap_destroy import HeatmapDestroy
# from .point_heatmap_destroy import HeatmapPointDestroy


class PointDestroy(LNSOperator):
    """Point based destroy operator.

    Select customers that are closest to a specific point and remove
    them from the solution.

    Args:
        percentage: The degree of destruction. A number in the range
            [0, 1].
        point: The reference point for destruction. If given, it will be
            constant throughout execution. Otherwise (default) it will
            be randomly sampled at each call.
    """

    def __init__(self, percentage: float,
                 point: Optional[Tuple[float, float]] = None):
        assert 0 <= percentage <= 1, ('Degree of destruction must be in the '
                                      'range [0, 1]')
        self.percentage = percentage
        self.point = point

        self.np_rng = nlns.numpy_generator_from_rng(self.rng)

    def set_random_state(self, seed: nlns.RandomSeedOrState):
        """Set internal random state for reproducible results.

        For this particular operator this means retrieving a ``Random``
        object and generating a ``np.random.Generator`` from its state.
        """
        super().set_random_state(seed)
        self.np_rng = nlns.numpy_generator_from_rng(self.rng)

    def apply_single(self, solution: VRPSolution) -> VRPSolution:
        """Apply the operator on a sinlge solution.

        Args:
            solution: The solution on which the operator is applied.

        Returns:
            The partially destroyed solution.
        """
        point = self.point
        if point is None:
            point = self.np_rng.random((1, 2))

        n = solution.instance.n_customers
        customers = np.array(solution.instance.customers)
        dist = np.sum((customers - point) ** 2, axis=1)
        closest_customers = np.argsort(dist)[:int(n * self.percentage)] + 1
        solution.destroy_nodes(to_remove=closest_customers)
        return solution

    def __call__(self, solutions: Sequence[VRPSolution]
                 ) -> Sequence[VRPSolution]:
        """Apply the operator.

        Args:
            solutions: The solutions on which the operator is applied.

        Returns:
            The partially destroyed solutions, indexed as the input
            ones.
        """
        return [self.apply_single(s) for s in solutions]


class RandomDestroy(LNSOperator):
    """Remove random customers from the given solutions.

    Args:
        percentage: The degree of destruction. A number in the range
            [0, 1].
    """

    def __init__(self, percentage: float):
        assert 0 <= percentage <= 1, ('Degree of destruction must be in the '
                                      'range [0, 1]')
        self.percentage = percentage

        self.np_rng = nlns.numpy_generator_from_rng(self.rng)

    def set_random_state(self, seed: nlns.RandomSeedOrState):
        """Set internal random state for reproducible results.

        For this particular operator this means retrieving a ``Random``
        object and generating a ``np.random.Generator`` from its state.
        """
        super().set_random_state(seed)
        self.np_rng = nlns.numpy_generator_from_rng(self.rng)

    def apply_single(self, solution: VRPSolution) -> VRPSolution:
        """Apply the operator on a sinlge solution.

        Args:
            solution: The solution on which the operator is applied.

        Returns:
            The partially destroyed solution.
        """
        n = solution.instance.n_customers
        random_customers = self.np_rng.choice(range(1, n + 1),
                                              int(n * self.percentage),
                                              replace=False)
        solution.destroy_nodes(to_remove=random_customers)
        return solution

    def __call__(self, solutions: Sequence[VRPSolution]) -> List[VRPSolution]:
        """Apply the operator.

        Args:
            solutions: The solutions on which the operator is applied.

        Returns:
            The partially destroyed solutions, indexed as the input
            ones.
        """
        return [self.apply_single(s) for s in solutions]


class TourDestroy(LNSOperator):
    """Tour based destroy operator.

    Select tours that are closest to a specific point and remove
    them from the solution.

    Args:
        percentage: The degree of destruction. A number in the range
            [0, 1].
        point: The reference point for destruction. If given, it will be
            constant throughout execution. Otherwise (default) it will
            be randomly sampled at each call.
    """

    def __init__(self, percentage: float, point: Tuple[float, float] = None):
        assert 0 <= percentage <= 1, ('Degree of destruction must be in the '
                                      'range [0, 1]')
        self.percentage = percentage
        self.point = point

        self.np_rng = nlns.numpy_generator_from_rng(self.rng)

    def set_random_state(self, seed: nlns.RandomSeedOrState):
        """Set internal random state for reproducible results.

        For this particular operator this means retrieving a ``Random``
        object and generating a ``np.random.Generator`` from its state.
        """
        super().set_random_state(seed)
        self.np_rng = nlns.numpy_generator_from_rng(self.rng)

    def __call__(self, solutions: Sequence[VRPSolution]) -> List[VRPSolution]:
        """Apply the operator.

        Args:
            solutions: The solutions on which the operator is applied.

        Returns:
            The partially destroyed solutions, indexed as the input
            ones.
        """
        return [self.apply_single(s) for s in solutions]

    def apply_single(self, solution: VRPSolution) -> VRPSolution:
        """Apply the operator on a sinlge solution.

        Args:
            solution: The solution on which the operator is applied.

        Returns:
            The partially destroyed solution.
        """
        point = self.point
        if point is None:
            point = self.np_rng.random((1, 2))

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
        # Number of customer that should be removed
        n_to_remove = int(n * self.percentage)
        n_removed = 0
        customers = np.array(solution.instance.customers)
        dist = np.sum((customers - point) ** 2, axis=1)
        closest_customers = np.argsort(dist) + 1
        # Iterate over customers starting with the customer closest
        # to the random point.
        for customer_idx in closest_customers:
            # Iterate over the tours of the customer
            for i in customer_to_tour[customer_idx]:
                # And if the tour is not yet marked for removal
                if i not in tours_to_remove:
                    # Mark it for removal
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
        return solution
