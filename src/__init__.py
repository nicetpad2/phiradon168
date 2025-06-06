"""Compatibility layer forwarding old `src` imports to `nicegold`."""
from importlib import import_module
import sys

_aliases = {
    'adaptive': 'adaptive',
    'config': 'config',
    'cooldown_utils': 'cooldown_utils',
    'dashboard': 'dashboard',
    'data_loader': 'data_loader',
    'evaluation': 'evaluation',
    'feature_analysis': 'feature_analysis',
    'features': 'features',
    'log_analysis': 'log_analysis',
    'main': 'main',
    'money_management': 'money_management',
    'monitor': 'monitor',
    'order_manager': 'order_manager',
    'qa_tools': 'qa_tools',
    'training': 'training',
    'wfv': 'wfv',
    'wfv_monitor': 'wfv_monitor',
    'strategy': 'strategy_core',
    'strategy_core': 'strategy_core',
    'strategy_pkg': 'strategy'
}

for old, new in _aliases.items():
    try:
        module = import_module(f'nicegold.{new}')
        sys.modules[f'src.{old}'] = module
    except ModuleNotFoundError:
        pass

def __getattr__(name):
    target = _aliases.get(name, name)
    return import_module(f'nicegold.{target}')
