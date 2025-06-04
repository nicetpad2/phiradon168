import importlib
import logging
import sys

from src.utils.env_utils import get_env_float


def test_get_env_float_valid(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "0.42")
    assert get_env_float("TEST_FLOAT", 0.1) == 0.42


def test_get_env_float_invalid(monkeypatch, caplog):
    monkeypatch.setenv("TEST_FLOAT", "abc")
    with caplog.at_level(logging.WARNING):
        assert get_env_float("TEST_FLOAT", 0.3) == 0.3
    assert "not a valid float" in caplog.text


def test_config_threshold_env(monkeypatch):
    monkeypatch.setenv("DRIFT_WASSERSTEIN_THRESHOLD", "0.25")
    if 'src.config' in sys.modules:
        monkeypatch.delitem(sys.modules, 'src.config', raising=False)
    cfg = importlib.import_module('src.config')
    assert cfg.DRIFT_WASSERSTEIN_THRESHOLD == 0.25
    monkeypatch.delenv('DRIFT_WASSERSTEIN_THRESHOLD', raising=False)
    importlib.reload(cfg)
