import pytest

from context import nlns
from nlns.generators import generate_multiple_instances
from nlns.instances import VRPSolution


@pytest.fixture
def set_default_rng():
    previous_state = nlns.default_rng.getstate()
    nlns.default_rng.seed(17170)
    yield
    nlns.default_rng.setstate(previous_state)


@pytest.fixture(params=[20, 50, 100])
def empty_solutions(request):
    # TODO: Handle seeds in generate_multiple_instances

    return tuple(map(VRPSolution,
                     generate_multiple_instances(50, request.param)))
