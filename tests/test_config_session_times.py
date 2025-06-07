import importlib
import os
import sys
import types

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))


def _import_config(monkeypatch):
    monkeypatch.setitem(sys.modules, 'seaborn', types.ModuleType('seaborn'))
    monkeypatch.setitem(sys.modules, 'requests', types.ModuleType('requests'))
    monkeypatch.setitem(sys.modules, 'shap', types.ModuleType('shap'))
    if 'src.config' in sys.modules:
        monkeypatch.delitem(sys.modules, 'src.config', raising=False)
    return importlib.import_module('src.config')


def test_session_times_valid(monkeypatch):
    monkeypatch.setenv('SESSION_TIMES_UTC', '{"Asia": [1, 8], "London": [7, 16]}')
    cfg = _import_config(monkeypatch)
    assert cfg.SESSION_TIMES_UTC == {'Asia': [1, 8], 'London': [7, 16]}


def test_session_times_invalid(monkeypatch, caplog):
    monkeypatch.setenv('SESSION_TIMES_UTC', 'oops')
    cfg = _import_config(monkeypatch)
    assert cfg.SESSION_TIMES_UTC == {'Asia': (22, 8), 'London': (7, 16), 'NY': (13, 21)}
    assert any('SESSION_TIMES_UTC env var invalid' in rec.message for rec in caplog.records)

