import os
import sys
import pandas as pd
import numpy as np
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from strategy import strategy as strategy_module


def test_apply_strategy_length_mismatch(monkeypatch):
    df = pd.DataFrame({"Close": [1, 2, 3]})
    monkeypatch.setattr(strategy_module, "generate_open_signals", lambda x: np.array([1, 0]))
    monkeypatch.setattr(strategy_module, "generate_close_signals", lambda x: np.array([0, 1, 0]))
    with pytest.raises(ValueError):
        strategy_module.apply_strategy(df)


def test_run_backtest_does_not_mutate():
    df = pd.DataFrame({"Close": [1.0, 1.1, 1.2]})
    original = df.copy()
    result = strategy_module.run_backtest(df, 1000.0)
    assert result == []
    pd.testing.assert_frame_equal(df, original)
