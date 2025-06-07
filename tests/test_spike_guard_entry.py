import os, sys
import pandas as pd
import numpy as np
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
from src import strategy

@pytest.mark.parametrize(
    "session,row,expected",
    [
        ("Asia", pd.Series({"spike_score": 0.9}), True),
        (
            "London",
            pd.Series({
                "spike_score": 0.9,
                "ADX": 30,
                "Wick_Ratio": 0.5,
                "Volatility_Index": 1.0,
                "Candle_Body": 0.3,
                "Candle_Range": 0.5,
                "Gain": 1.0,
                "ATR_14": 1.0,
            }),
            False,
        ),
        (
            "London",
            pd.Series({
                "spike_score": 0.1,
                "ADX": np.nan,
                "Wick_Ratio": 0.5,
                "Volatility_Index": 1.0,
                "Candle_Body": 0.3,
                "Candle_Range": 0.5,
                "Gain": 1.0,
                "ATR_14": 1.0,
            }),
            True,
        ),
        (
            "London",
            pd.Series({
                "spike_score": 0.1,
                "ADX": 10,
                "Wick_Ratio": 0.8,
                "Volatility_Index": 0.7,
                "Candle_Body": 0.3,
                "Candle_Range": 1.0,
                "Gain": 1.0,
                "ATR_14": 1.0,
            }),
            False,
        ),
        (
            "London",
            pd.Series({
                "spike_score": 0.1,
                "ADX": 30,
                "Wick_Ratio": 0.5,
                "Volatility_Index": 1.2,
                "Candle_Body": 1.0,
                "Candle_Range": 2.0,
                "Gain": 4.0,
                "ATR_14": 5.0,
            }),
            True,
        ),
    ],
)
def test_spike_guard_london(session, row, expected):
    assert strategy.spike_guard_london(row, session, 0) is expected

def make_row(vol, score):
    return pd.Series({
        "spike_score": 0.0,
        "ADX": 30,
        "Wick_Ratio": 0.5,
        "Volatility_Index": vol,
        "Candle_Body": 0.4,
        "Candle_Range": 0.8,
        "Gain": 1.0,
        "ATR_14": 1.0,
        "Signal_Score": score,
    })

@pytest.mark.parametrize(
    "row,expected_reason",
    [
        (make_row(0.5, 1.5), "LOW_VOLATILITY"),
        (make_row(1.2, np.nan), "INVALID_SIGNAL_SCORE"),
        (make_row(1.2, 0.2), "LOW_SIGNAL_SCORE"),
        (make_row(1.2, 1.5), "ALLOWED"),
    ],
)
def test_is_entry_allowed(monkeypatch, row, expected_reason):
    monkeypatch.setattr(strategy, "spike_guard_london", lambda *a, **k: True)
    allowed, reason = strategy.is_entry_allowed(
        row,
        "London",
        0,
        "BUY",
        "UP",
        signal_score_threshold=1.0,
    )
    assert ("ALLOWED" in reason) == (expected_reason == "ALLOWED")
    assert expected_reason.split("(")[0] in reason
