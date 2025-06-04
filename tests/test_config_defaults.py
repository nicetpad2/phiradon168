import importlib


def test_default_parameters():
    cfg = importlib.import_module('src.config')
    assert cfg.MIN_SIGNAL_SCORE_ENTRY == 1.0
    assert cfg.M15_TREND_RSI_UP == 51
    assert cfg.M15_TREND_RSI_DOWN == 49
    assert cfg.FORCED_ENTRY_MIN_GAIN_Z_ABS == 0.5
    assert set(["Normal", "Breakout", "StrongTrend", "Reversal", "InsideBar", "Choppy"]) == set(cfg.FORCED_ENTRY_ALLOWED_REGIMES)
