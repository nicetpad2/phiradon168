import importlib
import sys
import os
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)


def _reload_config():
    if 'config.config' in sys.modules:
        del sys.modules['config.config']
    return importlib.import_module('config.config')


def test_config_creates_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('MODEL_DIR', str(tmp_path / 'models'))
    monkeypatch.setenv('LOG_DIR', str(tmp_path / 'logs'))
    cfg_module = _reload_config()
    cfg = cfg_module.Config()
    assert cfg.DATA_DIR.is_dir()
    assert cfg.MODEL_DIR.is_dir()
    assert cfg.LOG_DIR.is_dir()


def test_config_numeric_env(monkeypatch):
    monkeypatch.setenv('NUM_WORKERS', '4')
    monkeypatch.setenv('LEARNING_RATE', '0.5')
    monkeypatch.setenv('LOG_DIR', 'tmp_logs')
    cfg_module = _reload_config()
    cfg = cfg_module.Config()
    assert cfg.NUM_WORKERS == 4
    assert isinstance(cfg.NUM_WORKERS, int)
    assert cfg.LEARNING_RATE == 0.5
    assert isinstance(cfg.LEARNING_RATE, float)


def test_config_invalid_num(monkeypatch):
    monkeypatch.setenv('NUM_WORKERS', 'abc')
    monkeypatch.setenv('LOG_DIR', 'tmp_logs')
    with pytest.raises(TypeError):
        _reload_config()


def test_config_invalid_float(monkeypatch):
    """Ensure invalid float env raises TypeError."""
    monkeypatch.setenv('LEARNING_RATE', 'oops')
    monkeypatch.setenv('LOG_DIR', 'tmp_logs')
    with pytest.raises(TypeError):
        _reload_config()
