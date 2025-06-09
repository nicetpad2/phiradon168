import importlib
import pytest


def test_default_parameters():
    cfg = importlib.import_module('src.config')
    assert cfg.MIN_SIGNAL_SCORE_ENTRY == pytest.approx(0.3, rel=1e-6)
    assert cfg.M15_TREND_RSI_UP == 60
    assert cfg.M15_TREND_RSI_DOWN == 40
    assert cfg.FORCED_ENTRY_MIN_GAIN_Z_ABS == 0.5
    assert set(["Normal", "Breakout", "StrongTrend", "Reversal", "InsideBar", "Choppy"]) == set(cfg.FORCED_ENTRY_ALLOWED_REGIMES)
    assert cfg.ENABLE_SOFT_COOLDOWN is True
    assert cfg.ADAPTIVE_SIGNAL_SCORE_QUANTILE == 0.4
    assert cfg.REENTRY_MIN_PROBA_THRESH == 0.40
    assert cfg.OMS_ENABLED is True
    assert cfg.OMS_DEFAULT is True
    assert cfg.PAPER_MODE is False
    assert cfg.POST_TRADE_COOLDOWN_BARS == 2
    # [Patch v5.9.3] New hyperparameter defaults
    assert cfg.LEARNING_RATE == pytest.approx(0.01, rel=1e-6)
    assert cfg.DEPTH == 6
    assert cfg.L2_LEAF_REG is None
    # [Patch v6.2.1] New placeholders and defaults
    assert cfg.SYMBOL == "XAUUSD"
    assert cfg.TIMEFRAME == "M1"
    for attr in [
        "subsample",
        "colsample_bylevel",
        "bagging_temperature",
        "random_strength",
        "seed",
    ]:
        assert hasattr(cfg, attr)
        assert getattr(cfg, attr) is None
