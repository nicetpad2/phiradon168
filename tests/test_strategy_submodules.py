import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from strategy.entry_rules import generate_open_signals
from strategy.exit_rules import generate_close_signals


def test_generate_open_signals_pkg():
    df = pd.DataFrame({"Close": [1.0, 1.2, 1.1], "MACD_hist": [0.1, 0.2, 0.3], "RSI": [60, 55, 40]})
    signals = generate_open_signals(df, use_macd=False, use_rsi=False)
    assert signals.tolist() == [0, 1, 0]


def test_generate_close_signals_pkg():
    df = pd.DataFrame({"Close": [1.0, 0.9, 1.1], "MACD_hist": [-0.1, -0.2, -0.3], "RSI": [40, 45, 60]})
    signals = generate_close_signals(df, use_macd=False, use_rsi=False)
    assert signals.tolist() == [0, 1, 0]
