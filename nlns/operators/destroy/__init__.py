from typing import Sequence, Tuple, Optional, List

import numpy as np

import nlns
from nlns.instances import VRPSolution
from nlns.operators import LNSOperator

# from .tour_destroy import TourDestroy
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
