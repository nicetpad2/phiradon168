import os
import sys
import types

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from src.utils import hardware


def test_has_gpu_true(monkeypatch):
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    assert hardware.has_gpu() is True
    monkeypatch.delitem(sys.modules, 'torch')


def test_has_gpu_false(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    assert hardware.has_gpu() is False
    monkeypatch.delitem(sys.modules, 'torch')


def test_estimate_resource_plan(monkeypatch):
    fake_psutil = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(total=8*1024**3))
    monkeypatch.setitem(sys.modules, 'psutil', fake_psutil)
    monkeypatch.setattr(hardware, 'has_gpu', lambda: False)
    plan = hardware.estimate_resource_plan()
    assert plan['n_folds'] >= 4 and plan['batch_size'] >= 16
    monkeypatch.delitem(sys.modules, 'psutil')


def test_estimate_plan_gpu(monkeypatch):
    """GPU available and device name resolved."""
    fake_psutil = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(total=16*1024**3))
    monkeypatch.setitem(sys.modules, 'psutil', fake_psutil)
    dummy_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda idx: "RTX"
        )
    )
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch)
    plan = hardware.estimate_resource_plan()
    assert plan['gpu'] == "RTX"
    monkeypatch.delitem(sys.modules, 'psutil')
    monkeypatch.delitem(sys.modules, 'torch')


def test_estimate_plan_gpu_name_error(monkeypatch):
    """GPU available but device name lookup fails."""
    fake_psutil = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(total=16*1024**3))
    monkeypatch.setitem(sys.modules, 'psutil', fake_psutil)
    dummy_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda idx: (_ for _ in ()).throw(RuntimeError("x"))
        )
    )
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch)
    plan = hardware.estimate_resource_plan()
    assert plan['gpu'] == "Unknown"
    monkeypatch.delitem(sys.modules, 'psutil')
    monkeypatch.delitem(sys.modules, 'torch')


def test_estimate_plan_psutil_missing(monkeypatch):
    """psutil import fails and defaults are returned."""
    import builtins
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'psutil':
            raise ImportError
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    monkeypatch.setattr(hardware, 'has_gpu', lambda: False)
    plan = hardware.estimate_resource_plan(default_folds=3, default_batch=15)
    assert plan == {'n_folds': 3, 'batch_size': 15, 'gpu': 'Unknown'}
    monkeypatch.setattr(builtins, '__import__', orig_import)
