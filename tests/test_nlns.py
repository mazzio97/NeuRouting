import pytest

from context import nlns


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
