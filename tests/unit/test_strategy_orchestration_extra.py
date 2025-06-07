import os
import sys
import pandas as pd
import numpy as np
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from strategy import strategy as strategy_module


def test_apply_strategy_empty_df(monkeypatch):
    df = pd.DataFrame({"Close": []})
    called = {"open": False, "close": False}

    def fake_open(arg):
        assert arg is df
        called["open"] = True
        return np.array([], dtype=np.int8)

    def fake_close(arg):
        assert arg is df
        called["close"] = True
        return np.array([], dtype=np.int8)

    monkeypatch.setattr(strategy_module, "generate_open_signals", fake_open)
    monkeypatch.setattr(strategy_module, "generate_close_signals", fake_close)

    res = strategy_module.apply_strategy(df)
    assert called == {"open": True, "close": True}
    assert res.empty and list(res.columns) == ["Close", "Entry", "Exit"]


def test_apply_strategy_overwrite(monkeypatch):
    df = pd.DataFrame({"Close": [1, 2], "Entry": [9, 9], "Exit": [8, 8]})
    monkeypatch.setattr(strategy_module, "generate_open_signals", lambda x: np.array([1, 0]))
    monkeypatch.setattr(strategy_module, "generate_close_signals", lambda x: np.array([0, 1]))

    result = strategy_module.apply_strategy(df)
    assert result["Entry"].tolist() == [1, 0]
    assert result["Exit"].tolist() == [0, 1]
    # Ensure original columns were not mutated
    assert df["Entry"].tolist() == [9, 9]
    assert df["Exit"].tolist() == [8, 8]


def test_run_backtest_empty_df():
    df = pd.DataFrame({"Close": []})
    assert strategy_module.run_backtest(df, 1000.0) == []


def test_run_backtest_negative_balance():
    df = pd.DataFrame({"Close": [1.0]})
    assert strategy_module.run_backtest(df, -100.0) == []
