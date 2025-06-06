import importlib


def test_default_parameters():
    cfg = importlib.import_module('src.config')
    assert cfg.MIN_SIGNAL_SCORE_ENTRY == 0.3
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
