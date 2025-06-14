import os
import sys
import types
import importlib
import subprocess

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))


def _import_config(monkeypatch, shap_exists=True):
    # Provide dummy dependencies so src.config import won't try to install
    dummy_ta = types.ModuleType('ta')
    dummy_ta.__version__ = '0.test'
    monkeypatch.setitem(sys.modules, 'ta', dummy_ta)
    monkeypatch.setitem(sys.modules, 'seaborn', types.ModuleType('seaborn'))
    monkeypatch.setitem(sys.modules, 'requests', types.ModuleType('requests'))
    if shap_exists:
        monkeypatch.setitem(sys.modules, 'shap', types.ModuleType('shap'))
    else:
        monkeypatch.delitem(sys.modules, 'shap', raising=False)
    if 'src.config' in sys.modules:
        monkeypatch.delitem(sys.modules, 'src.config', raising=False)
    return importlib.import_module('src.config')


def test_install_shap_noop_when_installed(monkeypatch):
    config = _import_config(monkeypatch, shap_exists=True)
    calls = []
    monkeypatch.setattr(config.subprocess, 'run', lambda *a, **k: calls.append(True))
    assert config.SHAP_INSTALLED is True
    config.install_shap()
    assert calls == []


def test_install_shap_failure(monkeypatch):
    """install_shap should handle installation errors gracefully."""
    monkeypatch.setattr(
        subprocess,
        'run',
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('fail')),
    )

    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'shap':
            raise ImportError('no shap')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    config = _import_config(monkeypatch, shap_exists=False)
    assert config.SHAP_INSTALLED is False
    assert config.SHAP_AVAILABLE is False
    assert getattr(config, 'shap', None) is None

    monkeypatch.setattr(config.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('fail')))
    config.install_shap()
    assert config.SHAP_AVAILABLE is False
    assert getattr(config, 'shap', None) is None


def test_file_base_override(monkeypatch, tmp_path):
    monkeypatch.setenv('FILE_BASE_OVERRIDE', str(tmp_path))
    config = _import_config(monkeypatch)
    assert config.FILE_BASE == str(tmp_path)


def test_file_base_mount_failure_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv('FILE_BASE_OVERRIDE', str(tmp_path/'nope'))
    monkeypatch.setenv('COLAB_RELEASE_TAG', '1')
    dummy_colab = types.ModuleType('google.colab')
    dummy_colab.drive = types.SimpleNamespace(mount=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    google_mod = types.ModuleType('google')
    google_mod.colab = dummy_colab
    monkeypatch.setitem(sys.modules, 'google', google_mod)
    monkeypatch.setitem(sys.modules, 'google.colab', dummy_colab)
    ip_module = types.ModuleType('IPython')
    ip_module.get_ipython = lambda: types.SimpleNamespace(kernel=object())
    monkeypatch.setitem(sys.modules, 'IPython', ip_module)
    config = _import_config(monkeypatch)
    expected = os.path.abspath(os.path.join(os.path.dirname(config.__file__), '..'))
    assert config.FILE_BASE == expected


def test_file_base_detect_drive(monkeypatch):
    """FILE_BASE should point to Google Drive path when available."""
    monkeypatch.delenv('FILE_BASE_OVERRIDE', raising=False)
    monkeypatch.delenv('COLAB_RELEASE_TAG', raising=False)
    monkeypatch.delenv('COLAB_GPU', raising=False)
    # Patch os.path.isdir to simulate Drive folder present
    original_isdir = os.path.isdir
    def fake_isdir(path):
        if path == '/content/drive/MyDrive/Phiradon168':
            return True
        return original_isdir(path)
    monkeypatch.setattr(os.path, 'isdir', fake_isdir)
    config = _import_config(monkeypatch)
    assert config.FILE_BASE == '/content/drive/MyDrive/Phiradon168'
