import importlib


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
