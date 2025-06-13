import importlib
import types
import sys
import os
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))


def test_ensure_ta_installed(monkeypatch, caplog):
    dummy_ta = types.ModuleType('ta')
    dummy_ta.__version__ = '0.test'
    monkeypatch.setitem(sys.modules, 'ta', dummy_ta)
    monkeypatch.setitem(sys.modules, 'seaborn', types.ModuleType('seaborn'))
    monkeypatch.setitem(sys.modules, 'requests', types.ModuleType('requests'))
    monkeypatch.setitem(sys.modules, 'shap', types.ModuleType('shap'))
    if 'src.config' in sys.modules:
        monkeypatch.delitem(sys.modules, 'src.config', raising=False)
    logger = logging.getLogger("NiceGold")
    captured = []
    monkeypatch.setattr(logger, "info", lambda msg, *a, **k: captured.append(msg))
    config = importlib.import_module('src.config')
    assert config.TA_VERSION == '0.test'
    assert any('Using TA version: 0.test' in m for m in captured)


def test_ta_version_fallback(monkeypatch, caplog):
    dummy_ta = types.ModuleType('ta')
    monkeypatch.setitem(sys.modules, 'ta', dummy_ta)
    monkeypatch.setitem(sys.modules, 'seaborn', types.ModuleType('seaborn'))
    monkeypatch.setitem(sys.modules, 'requests', types.ModuleType('requests'))
    monkeypatch.setitem(sys.modules, 'shap', types.ModuleType('shap'))
    original_version = importlib.metadata.version
    def fake_version(name):
        if name == 'ta':
            raise Exception('not installed')
        return original_version(name)
    monkeypatch.setattr(importlib.metadata, 'version', fake_version)
    if 'src.config' in sys.modules:
        monkeypatch.delitem(sys.modules, 'src.config', raising=False)
    logger = logging.getLogger("NiceGold")
    captured = []
    monkeypatch.setattr(logger, "info", lambda msg, *a, **k: captured.append(msg))
    config = importlib.import_module('src.config')
    assert config.TA_VERSION == 'N/A'
    assert any('Using TA version: N/A' in m for m in captured)
