import importlib
from src import strategy, config


def test_rsi_drift_override_threshold_default():
    assert strategy.DEFAULT_RSI_DRIFT_OVERRIDE_THRESHOLD == 0.65
    assert config.RSI_DRIFT_OVERRIDE_THRESHOLD == 0.65
