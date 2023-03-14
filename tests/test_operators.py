import pytest

from context import nlns
from nlns.operators import LNSOperator
from helpers import set_default_rng                         # NOQA


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
