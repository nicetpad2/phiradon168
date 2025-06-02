import importlib
import types
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest


def test_entry_config_import_error(monkeypatch):
    # Import src.main normally; config import should fail due to missing deps
    if 'src.main' in sys.modules:
        del sys.modules['src.main']
    main = importlib.import_module('src.main')
    assert main.DEFAULT_ENTRY_CONFIG_PER_FOLD == {}


def test_entry_config_loaded_from_config(monkeypatch):
    dummy = types.ModuleType('config')
    dummy.ENTRY_CONFIG_PER_FOLD = {'x': 1}
    monkeypatch.setitem(sys.modules, 'src.config', dummy)
    if 'src.main' in sys.modules:
        del sys.modules['src.main']
    main = importlib.import_module('src.main')
    assert main.DEFAULT_ENTRY_CONFIG_PER_FOLD == {'x': 1}
