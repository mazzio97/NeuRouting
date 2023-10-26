import importlib
import random
from typing import Union, Tuple

import numpy as np

default_rng = random.Random()
RandomSeedOrState = Union[None, int, float, str, bytes, bytearray, Tuple]


def module_found(module_name: str, current_module_name: str,
                 hard_requirement: bool = True):
    """Check if a module is importable, raise an exception otherwise.

    Used by modules that rely on third party libraries in order to
    throw a clean exception that describes a missing extra dependency.

    Args:
        module_name: The name of the required module. If it is a
            submodule/subpackage, use dot notation.
        current_module_name: The name of the current module, which
            requires ``module_name``. Usually, pass ``__name__``.
        hard_requirement: Whether the module is a hard requirement. A
            hard requirement will always throw an exception as it is
            mandatory. Non-hard (soft) requirements will only raise
            a warning (TODO).

    Raises:
        ModuleNotFoundError: If the module could not be imported. The
            message shall explain that it is an optional dependency not
            met.
    """
    found = True
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        found = False

    # Raise outside the except block to get a cleaner traceback
    if not found:
        raise ModuleNotFoundError(
            f'Module/package named "{module_name}" '
            f'could not be found. {current_module_name} module '
            'requires it to work. Consider installing it.')


def get_rng(seed: RandomSeedOrState = None) -> random.Random:
    """Generate a random number generator based on input state.

    Args:
        seed: A seed of type ``int``, ``float``, ``str``,
            ``bytes`` or ``bytearray``, or a random state (a ``Tuple``
            obtained via ``random.getstate()`` or similar).
            In the latter case, a generator is created and its
            state is set to the given one.
            If not provided (``None`` value), the default random
            generator is returned (:attr:`default_rng`).
    """
    if seed is None:
        return default_rng

    new_rng = random.Random()

    # Simple heuristic for random states
    if type(seed) is tuple and len(seed) == 3:
        new_rng.setstate(seed)
        return new_rng

    new_rng.seed(seed)
    return new_rng


def numpy_generator_from_rng(rng: random.Random) -> np.random.Generator:
    """Build a numpy rng from a ``random.Random`` one.

    Input generator is not consumed, meaning that multiple calls with
    the same (untouched) input will result in the same numpy generator
    state.

    Args:
        rng: A Python standard random generator.

    Returns:
        A numpy generator built from the state of input generator.
    """
    return np.random.default_rng(rng.getstate()[1])


# Import here some modules for convenience
from . import io                            # NOQA
