import random

import pytest

from context import nlns


@pytest.fixture
def set_default_rng():
    previous_state = nlns.default_rng.getstate()
    nlns.default_rng.seed(17170)
    yield
    nlns.default_rng.setstate(previous_state)


def test_tests():
    assert True


@pytest.mark.parametrize('module_name, exception',
                         [('random', None), ('xxxxxx', ModuleNotFoundError)])
def test_module_found(module_name, exception, current_module='yyyyyy'):
    if exception is None:
        nlns.module_found(module_name, current_module)
        return

    with pytest.raises(exception):
        nlns.module_found(module_name, current_module)


@pytest.mark.parametrize('seed', [1, 1., 'x', b'xy', bytearray(b'xyz'),
                                  random.Random(88).getstate()])
def test_get_rng(seed, set_default_rng):
    rng = nlns.get_rng(seed)
    assert rng.getstate() != nlns.default_rng.getstate()


def test_get_rng_none(set_default_rng):
    assert nlns.get_rng() is nlns.default_rng
