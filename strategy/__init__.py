"""Compatibility stub for old 'strategy' imports."""
from importlib import import_module
import sys

module = import_module('nicegold.strategy')
sys.modules[__name__] = module
