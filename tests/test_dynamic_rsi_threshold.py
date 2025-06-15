import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.feature_analysis import get_dynamic_rsi_threshold, suggest_rsi_thresholds_by_session
from strategy.entry_rules import generate_open_signals


def test_get_dynamic_rsi_threshold_basic():
    assert get_dynamic_rsi_threshold("NY", 1.2) == 45.0
    assert get_dynamic_rsi_threshold("Asia", 0.6) == 55.0
    assert get_dynamic_rsi_threshold("London", 0.9) == 50.0


def test_suggest_rsi_thresholds_by_session():
    df = pd.DataFrame({
        "session": ["NY", "NY", "Asia", "London"],
        "Volatility_Index": [1.2, 0.8, 0.6, 1.0],
    })
    res = suggest_rsi_thresholds_by_session(df)
    assert res["NY"] == 45.0
    assert res["Asia"] == 55.0
    assert res["London"] == 50.0


def test_generate_open_signals_dynamic_rsi():
    df = pd.DataFrame({
        "Close": [1, 1.1, 1.2],
        "MACD_hist": [0.1, 0.1, 0.1],
        "RSI": [46, 56, 51],
        "MA_fast": [1, 1, 1],
        "MA_slow": [0, 0, 0],
        "session": ["NY", "Asia", "London"],
        "Volatility_Index": [1.2, 0.6, 1.0],
    })
    res = generate_open_signals(df, use_macd=False, use_rsi=True)
    assert res.tolist() == [0, 1, 1]
