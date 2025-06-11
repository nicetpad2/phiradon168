import os
import importlib
import pandas as pd
from src import entry_rules
from src import config


def test_env_override(monkeypatch):
    monkeypatch.setenv("MIN_SIGNAL_SCORE_ENTRY", "0.8")
    df = pd.DataFrame({"signal_score": [0.9], "close": [1.0]})
    sig = entry_rules.generate_open_signals(df)
    assert sig.iloc[0] == 1
    monkeypatch.delenv("MIN_SIGNAL_SCORE_ENTRY", raising=False)


def test_ma_fallback(monkeypatch):
    monkeypatch.setenv("MIN_SIGNAL_SCORE_ENTRY", "0.8")
    df = pd.DataFrame(
        {
            "signal_score": [0.1] * 5,
            "close": [5, 4, 3, 4, 5],
        }
    )
    importlib.reload(entry_rules)
    monkeypatch.setattr(entry_rules.config, "FAST_MA_PERIOD", 2, raising=False)
    monkeypatch.setattr(entry_rules.config, "SLOW_MA_PERIOD", 3, raising=False)
    sig = entry_rules.generate_open_signals(df)
    assert sig.iloc[-1] == 1
    monkeypatch.delenv("MIN_SIGNAL_SCORE_ENTRY", raising=False)
