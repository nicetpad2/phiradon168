import os
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib


def _import_config(monkeypatch):
    # Provide dummy seaborn to satisfy import
    monkeypatch.setitem(sys.modules, 'seaborn', types.ModuleType('seaborn'))
    monkeypatch.setitem(sys.modules, 'requests', types.ModuleType('requests'))
    monkeypatch.setitem(sys.modules, 'shap', types.ModuleType('shap'))
    if 'src.config' in sys.modules:
        monkeypatch.delitem(sys.modules, 'src.config', raising=False)
    return importlib.import_module('src.config')


def test_is_colab_false(monkeypatch):
    if 'google.colab' in sys.modules:
        monkeypatch.delitem(sys.modules, 'google.colab', raising=False)
    config = _import_config(monkeypatch)
    assert config.is_colab() is False


def test_is_colab_true(monkeypatch):
    dummy = types.ModuleType('google.colab')
    dummy.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
    parent = types.ModuleType('google')
    parent.colab = dummy
    monkeypatch.setitem(sys.modules, 'google', parent)
    monkeypatch.setitem(sys.modules, 'google.colab', dummy)
    config = _import_config(monkeypatch)
    assert config.is_colab() is True
