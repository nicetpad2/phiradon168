"""Helper utilities for dynamic module management."""

from types import ModuleType
import importlib
import sys

# [Patch v6.3.3] Provide safe_reload to avoid ImportError when module removed

def safe_reload(module: ModuleType) -> ModuleType:
    """Reload a module even if it was removed from ``sys.modules``.

    Parameters
    ----------
    module : ModuleType
        Module object to reload.

    Returns
    -------
    ModuleType
        Reloaded module reference.
    """
    if not isinstance(module, ModuleType):
        raise TypeError("module must be a module object")

    name = getattr(getattr(module, "__spec__", None), "name", None) or getattr(module, "__name__", None)
    if name is None:
        raise TypeError("unable to determine module name")

    if sys.modules.get(name) is not module:
        module = importlib.import_module(name)

    return importlib.reload(module)
