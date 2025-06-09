"""Top-level package for project modules with lazy imports."""

from __future__ import annotations

import importlib
from types import ModuleType

# Modules to expose when importing `src` directly. They will be imported on first
# attribute access to avoid heavy dependencies at import time.
_EXPOSED_MODULES = {
    'adaptive',
    'money_management',
    'evaluation',
    'wfv',
    'param_stability',
    'strategy',
    'features',
    'data_loader',
    'training',
    'config',
    'qa_tools',
    'log_analysis',
    'dashboard',
    'realtime_dashboard',
    'utils',
}

__all__ = sorted(_EXPOSED_MODULES)


def __getattr__(name: str) -> ModuleType:
    """Dynamically import submodules listed in ``_EXPOSED_MODULES``."""
    if name in _EXPOSED_MODULES:
        module = importlib.import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

