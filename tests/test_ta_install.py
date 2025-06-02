import importlib
import types
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))


def test_ensure_ta_installed(monkeypatch):
    dummy_ta = types.ModuleType('ta')
    dummy_ta.__version__ = '0.test'
    monkeypatch.setitem(sys.modules, 'ta', dummy_ta)
    monkeypatch.setitem(sys.modules, 'seaborn', types.ModuleType('seaborn'))
    monkeypatch.setitem(sys.modules, 'requests', types.ModuleType('requests'))
    monkeypatch.setitem(sys.modules, 'shap', types.ModuleType('shap'))
    if 'src.config' in sys.modules:
        monkeypatch.delitem(sys.modules, 'src.config', raising=False)
    config = importlib.import_module('src.config')
    assert config.TA_VERSION == '0.test'
