import importlib
import os
import sys
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)


def _reload_config(monkeypatch):
    if 'config.config' in sys.modules:
        del sys.modules['config.config']
    return importlib.import_module('config.config')


def test_min_trade_rows_default(monkeypatch, tmp_path):
    monkeypatch.delenv('MIN_TRADE_ROWS', raising=False)
    monkeypatch.setenv('LOG_DIR', str(tmp_path / 'logs'))
    cfg_module = _reload_config(monkeypatch)
    cfg = cfg_module.Config()
    assert cfg.MIN_TRADE_ROWS == 10


def test_min_trade_rows_invalid(monkeypatch, tmp_path):
    monkeypatch.setenv('MIN_TRADE_ROWS', '0')
    monkeypatch.setenv('LOG_DIR', str(tmp_path / 'logs'))
    with pytest.raises(SystemExit):
        _reload_config(monkeypatch)
