import copy
from operator import attrgetter

import pytest

from context import nlns
from nlns.operators import LNSOperator
from nlns.operators.repair import GreedyRepair
from helpers import set_default_rng, empty_solutions        # NOQA


class IdentityOperator(LNSOperator):
    """Return solutions unchanged. Testing purposes."""

    def __call__(self, solutions):
        return solutions


@pytest.fixture
def identity_operator():
    return IdentityOperator()


class TestLNSOperator:

    def test_init(self, identity_operator, set_default_rng):    # NOQA
        assert identity_operator.rng is nlns.default_rng

    def test_set_random_state(self, identity_operator,
                              set_default_rng, seed=1):         # NOQA
        identity_operator.set_random_state(seed)
        assert identity_operator.rng.getstate() != nlns.default_rng.getstate()


class TestRepair:

    @pytest.mark.parametrize('operator_type, seed', [(GreedyRepair, 42)])
    def test_repair(self, operator_type, seed, empty_solutions):        # NOQA
        operator = operator_type()

        operator.set_random_state(seed)
        solutions = operator(empty_solutions)

        assert all(not solution.missing_customers() for solution in solutions)

    @pytest.mark.parametrize('operator_type, seed', [(GreedyRepair, 42)])
    def test_reproducibility(self, operator_type, seed,
                             empty_solutions):                  # NOQA
        operator = operator_type()

        empty_copy = copy.deepcopy(empty_solutions)

        operator.set_random_state(seed)
        solutions = operator(empty_solutions)

        operator.set_random_state(seed)
        solutions_copy = operator(empty_copy)

        costs = list(map(attrgetter('cost'), solutions))
        costs_copy = list(map(attrgetter('cost'), solutions_copy))

        assert costs == costs_copy
