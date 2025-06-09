# [Patch] package marker for src/
# Enables imports like `import src.config` under pytest
"""Top-level package for project modules with lazy imports.

This module exposes frequently used submodules while avoiding heavy
imports during package initialization. Submodules are imported on first
access via ``__getattr__`` to keep test environments lightweight.
"""

from __future__ import annotations

import importlib
import sys

# List of submodules that can be lazily loaded via ``from src import <module>``.
_SUBMODULES = [
    "adaptive",
    "money_management",
    "evaluation",
    "wfv",
    "param_stability",
    "config",
    "data_loader",
    "feature_analysis",
    "features",
    "main",
    "order_manager",
    "realtime_dashboard",
    "signal_classifier",
    "strategy",
    "training",
    "wfv_monitor",
    "log_analysis",
    "qa_tools",
]

__all__ = list(_SUBMODULES)


def __getattr__(name: str):
    """Dynamically import submodules on first access."""
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        setattr(sys.modules[__name__], name, module)
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

