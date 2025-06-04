import types
import sys
from src.utils import hardware


def test_has_gpu_true(monkeypatch):
    dummy = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, 'torch', dummy)
    assert hardware.has_gpu() is True


def test_has_gpu_false(monkeypatch):
    dummy = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, 'torch', dummy)
    assert hardware.has_gpu() is False


def test_has_gpu_no_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    assert hardware.has_gpu() is False
