import builtins
import importlib
import logging
import os
import sys
import types

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


def _import_config(monkeypatch):
    monkeypatch.setitem(sys.modules, 'seaborn', types.ModuleType('seaborn'))
    monkeypatch.setitem(sys.modules, 'requests', types.ModuleType('requests'))
    if 'src.config' in sys.modules:
        monkeypatch.delitem(sys.modules, 'src.config', raising=False)
    return importlib.import_module('src.config')


def test_mkl_error_fallback(monkeypatch, caplog):
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'torch':
            raise RuntimeError('Intel MKL FATAL ERROR: cannot load mkl_intel_thread.dll')
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    with caplog.at_level(logging.WARNING):
        cfg = _import_config(monkeypatch)
    assert cfg.USE_GPU_ACCELERATION is False
    assert any('GPU acceleration disabled due to import error' in m for m in caplog.messages)
