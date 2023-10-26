from abc import abstractmethod, ABC
from typing import Sequence

import nlns
from nlns.instances import VRPSolution


class LNSOperator(ABC):
    """Base class for large neighborhood search operators.

    Can be used with :class:`nlns.search.BaseLargeNeighborhoodSearch`
    subclasses (e.g. :class:`nlns.search.LNS`).

    The interface is the same for both destroy and repair operators.
    Minimal usage requires to implement :meth:`__call__` (see docs of
    the method for more info).

    When designing an operator, special care should go towards the
    capability of providing reproducible results. By default, the
    :attr:`rng` attribute provides a randomly initialized state which
    can be easily managed via :meth:`set_random_state`. When interacting
    with third party libraries, a custom implementation may be needed
    (see :meth:`set_random_state` for more).
    """
    rng = nlns.default_rng

    def set_random_state(self, seed: nlns.RandomSeedOrState):
        """Set internal random state for reproducible results.

        Default implementation retrieves a ``random.Random`` object
        through :func:`nlns.get_rng` and sets it to :attr:`rng`.

        Override to provide custom state generation (e.g. for third
        party libraries). The developer is responsible for using
        such state (both the provided and custom ones) at inference
        time.

        Args:
            seed: Random seed or state (see :func:`nlns.get_rng`).
        """
        self.rng = nlns.get_rng(seed)

    @abstractmethod
    def __call__(self,
                 solutions: Sequence[VRPSolution]) -> Sequence[VRPSolution]:
        """Abstract: repair or destroy sequence of solutions.

        Override to provide custom implementation of a destroy or repair
        operator.

        Args:
            solutions: A copy of the solutions to apply the operator on.
                As copies, they can be modified in place by the
                operator.
        Returns:
            Repaired or destroyed solutions, following the indexing of
                the input ones.
        """
