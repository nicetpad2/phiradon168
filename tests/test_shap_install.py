import importlib
import sys
import os
import types

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))


def _import_config(monkeypatch):
    monkeypatch.setitem(sys.modules, 'seaborn', types.ModuleType('seaborn'))
    monkeypatch.setitem(sys.modules, 'requests', types.ModuleType('requests'))
    if 'src.config' in sys.modules:
        monkeypatch.delitem(sys.modules, 'src.config', raising=False)
    return importlib.import_module('src.config')


def test_install_shap_sets_flags(monkeypatch):
    if 'shap' in sys.modules:  # pragma: no cover - optional cleanup
        monkeypatch.delitem(sys.modules, 'shap', raising=False)
    config = _import_config(monkeypatch)
    dummy_shap = types.ModuleType('shap')
    dummy_shap.__version__ = '0.test'

    def fake_run(*args, **kwargs):
        sys.modules['shap'] = dummy_shap
        return types.SimpleNamespace(stdout='ok')

    monkeypatch.setattr(config.subprocess, 'run', fake_run)
    config.install_shap()
    assert config.SHAP_INSTALLED is True
    assert config.SHAP_AVAILABLE is True


def test_use_gpu_env_var(monkeypatch):
    monkeypatch.setenv('USE_GPU_ACCELERATION', 'yes')
    monkeypatch.setitem(sys.modules, 'shap', types.ModuleType('shap'))
    dummy_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda idx: 'GPU'
        )
    )
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch)
    config = _import_config(monkeypatch)
    assert config.USE_GPU_ACCELERATION is True

def test_print_gpu_utilization_no_psutil(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, 'shap', types.ModuleType('shap'))
    config = _import_config(monkeypatch)
    monkeypatch.setattr(config, 'psutil', None, raising=False)
    monkeypatch.setattr(config, 'USE_GPU_ACCELERATION', False, raising=False)
    config.print_gpu_utilization('T')

