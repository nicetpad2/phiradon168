import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, "src"))

import src.strategy as strategy


def test_generate_open_signals_toggle_off():
    df = pd.DataFrame({"Close": [1.0, 1.2, 1.1], "MACD_hist": [0.1, 0.2, 0.3], "RSI": [60, 55, 40]})
    signals = strategy.generate_open_signals(df, use_macd=False, use_rsi=False)
    assert signals.tolist() == [0, 0, 0]


def test_generate_open_signals_with_indicators():
    df = pd.DataFrame({"Close": [1.0, 1.2, 1.3], "MACD_hist": [0.2, 0.1, -0.1], "RSI": [60, 55, 40]})
    signals = strategy.generate_open_signals(df, use_macd=True, use_rsi=True)
    assert signals.tolist() == [0, 1, 0]


def test_generate_close_signals_toggle_off():
    df = pd.DataFrame({"Close": [1.0, 0.9, 1.1], "MACD_hist": [-0.1, -0.2, -0.3], "RSI": [40, 45, 60]})
    signals = strategy.generate_close_signals(df, use_macd=False, use_rsi=False)
    assert signals.tolist() == [0, 1, 0]


def test_generate_close_signals_with_indicators():
    df = pd.DataFrame({"Close": [1.0, 0.9, 0.8], "MACD_hist": [-0.1, -0.2, 0.1], "RSI": [40, 45, 60]})
    signals = strategy.generate_close_signals(df, use_macd=True, use_rsi=True)
    assert signals.tolist() == [0, 1, 0]


def test_generate_open_signals_with_trend_and_volume():
    df = pd.DataFrame({
        "Close": [1.0, 0.9, 1.2],
        "RSI": [55, 55, 55],
        "Volume": [100, 100, 200],
        "MA_fast": [1.0, 1.0, 1.1],
        "MA_slow": [1.0, 1.0, 1.05],
    })
    signals = strategy.generate_open_signals(
        df,
        use_macd=False,
        use_rsi=True,
        trend="UP",
        vol_window=2,
    )
    assert signals.tolist() == [0, 0, 1]
