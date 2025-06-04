import importlib
import logging
import sys

from src.utils.env_utils import get_env_float
import pytest


def test_get_env_float_valid(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "0.42")
    assert get_env_float("TEST_FLOAT", 0.1) == 0.42


def test_get_env_float_invalid(monkeypatch, caplog):
    monkeypatch.setenv("TEST_FLOAT", "abc")
    with caplog.at_level(logging.ERROR):
        assert get_env_float("TEST_FLOAT", 0.3) == 0.3
    assert "cannot be parsed as float" in caplog.text


def test_get_env_float_missing(monkeypatch, caplog):
    monkeypatch.delenv("TEST_FLOAT", raising=False)
    with caplog.at_level(logging.INFO):
        assert get_env_float("TEST_FLOAT", 0.5) == 0.5
    assert "TEST_FLOAT not set, using default 0.5" in caplog.text


def test_get_env_float_scientific(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "1e-3")
    assert get_env_float("TEST_FLOAT", 0.1) == 0.001


def test_get_env_float_key_type():
    with pytest.raises(TypeError):
        get_env_float(None, 0.1)


def test_config_threshold_env(monkeypatch):
    monkeypatch.setenv("DRIFT_WASSERSTEIN_THRESHOLD", "0.25")
    if 'src.config' in sys.modules:
        monkeypatch.delitem(sys.modules, 'src.config', raising=False)
    cfg = importlib.import_module('src.config')
    assert cfg.DRIFT_WASSERSTEIN_THRESHOLD == 0.25
    monkeypatch.delenv('DRIFT_WASSERSTEIN_THRESHOLD', raising=False)
    importlib.reload(cfg)
