import builtins
import importlib
import runpy
import types
import sys
import logging
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)


def test_projectp_import_without_pynvml(monkeypatch):
    """Module should handle missing pynvml gracefully."""
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'pynvml':
            raise ImportError('no nvml')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    monkeypatch.delitem(sys.modules, 'ProjectP', raising=False)
    module = importlib.import_module('ProjectP')
    assert module.pynvml is None
    assert module.nvml_handle is None


def test_projectp_main_logs_gpu_status(monkeypatch, caplog):
    """Running the script should log GPU availability."""
    dummy_main = lambda: None
    monkeypatch.setitem(sys.modules, 'src.main', types.SimpleNamespace(main=dummy_main))
    monkeypatch.setattr(sys, 'argv', ['ProjectP.py'])
    with caplog.at_level(logging.INFO):
        runpy.run_path('ProjectP.py', run_name='__main__')
    assert any('GPU not available' in m for m in caplog.messages)
