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
