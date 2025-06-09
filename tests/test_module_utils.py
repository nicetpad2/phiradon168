import importlib
import sys
import types

from src.utils.module_utils import safe_reload


def test_safe_reload_imports_if_missing(monkeypatch):
    dummy = types.ModuleType('dummy_mod')
    dummy.__spec__ = types.SimpleNamespace(name='dummy_mod')
    if 'dummy_mod' in sys.modules:
        del sys.modules['dummy_mod']

    def fake_import(name):
        sys.modules[name] = dummy
        return dummy

    monkeypatch.setattr(importlib, 'import_module', fake_import)
    reloaded = safe_reload(dummy)
    assert reloaded is dummy
    assert sys.modules['dummy_mod'] is dummy
