import importlib

import pytest

from context import nlns
from nlns.generators import generate_multiple_instances
from nlns.instances import VRPSolution
from nlns.operators.initial import nearest_neighbor_solution


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


@pytest.fixture(params=[20, 50, 100])
def complete_solutions(request):
    # TODO: Handle seeds in generate_multiple_instances

    return tuple(map(nearest_neighbor_solution,
                     generate_multiple_instances(50, request.param)))


def skipif_module(required_module_name):
    """Mark to skip if the given module name cannot be imported."""
    found = True
    try:
        importlib.import_module(required_module_name)
    except (ModuleNotFoundError, ImportError):
        found = False

    return pytest.mark.skipif(
        not found,
        reason=f'Could not import module {required_module_name}')


class MissingPlaceholder:
    """Use as placeholder for missing imports (instead of None)."""

    def __init__(self, name: str):
        self.__name__ = name
