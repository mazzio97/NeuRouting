from typing import Mapping, Sequence

import numpy as np

import nlns
from nlns.operators.destroy import PointDestroy
from nlns.operators import LNSOperator
from nlns.instances import VRPInstance, VRPSolution


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numpy based softmax."""
    shifted_x = x - x.max()
    numerator = np.exp(shifted_x)
    return numerator / numerator.sum()


class PointHeatmapDestroy(LNSOperator):
    """ """

    def __init__(self, percentage: float,
                 heatmaps_map: Mapping[VRPInstance, np.ndarray]):
        assert 0 <= percentage <= 1, ('Degree of destruction must be in the '
                                      'range [0, 1]')
        self.percentage = percentage
        self.heatmaps_map = heatmaps_map

        self.np_rng = nlns.numpy_generator_from_rng(self.rng)

    def set_random_state(self, seed: nlns.RandomSeedOrState):
        """Set internal random state for reproducible results.

        For this particular operator this means retrieving a ``Random``
        object and generating a ``np.random.Generator`` from its state.
        """
        super().set_random_state(seed)
        self.np_rng = nlns.numpy_generator_from_rng(self.rng)

    def destroy_single(self, solution: VRPSolution) -> VRPSolution:
        """Destroy a single given solution."""
        instance = solution.instance
        heatmap = 1 - self.heatmaps_map[instance]
        # Mask (with a very negative value) edges which are not in the
        # examined solution
        solution_heatmap = _softmax(
            (heatmap + -1e12 * (solution.adjacency_matrix() == 0)).flatten())
        # Sample to find a likely low-probable edge
        sampled_edge, = self.rng.choices(range(heatmap.size),
                                         solution_heatmap)
        from_node, to_node = np.unravel_index(sampled_edge, heatmap.shape)

        # Instance customers are not guaranteed to be stored in a numpy
        # array, so we manually compute the average here, which should
        # still be more efficient than allocating new numpy arrays to do
        # it. On the other hand, the PointDestroy implementation used
        # right below ignores such considerations and allocates a lot of
        # random stuff.
        # -1 to the node ids to account for the depot
        from_position = instance.customers[from_node - 1]
        if from_node == 0:
            from_position = instance.depot
        to_position = instance.customers[to_node - 1]
        if to_node == 0:
            to_position = instance.depot
        destroy_point = ((from_position[0] + to_position[0]) / 2,
                         (from_position[1] + to_position[1]) / 2)

        # Use the existing point destroy implementation
        point_operator = PointDestroy(self.percentage, destroy_point)
        point_operator.np_rng = self.np_rng
        point_operator.rng = self.rng

        return point_operator.apply_single(solution)

    def __call__(self,
                 solutions: Sequence[VRPSolution]) -> Sequence[VRPSolution]:
        return [self.destroy_single(solution) for solution in solutions]
