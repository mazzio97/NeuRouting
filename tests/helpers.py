import pytest

from context import nlns


@pytest.fixture
def set_default_rng():
    previous_state = nlns.default_rng.getstate()
    nlns.default_rng.seed(17170)
    yield
    nlns.default_rng.setstate(previous_state)
