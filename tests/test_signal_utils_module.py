import os
import sys
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, "src"))

import src.signal_utils as sig_util


def test_open_close_signals_basic():
    df = pd.DataFrame({
        "Close": [1.0, 0.9, 1.1],
        "MACD_hist": [0.1, 0.2, 0.3],
        "RSI": [60, 55, 40],
    })
    open_sig = sig_util.generate_open_signals(df, use_macd=False, use_rsi=False)
    close_sig = sig_util.generate_close_signals(df, use_macd=False, use_rsi=False)
    assert open_sig.tolist() == [0, 0, 0]
    assert close_sig.tolist() == [0, 1, 0]


def test_precompute_arrays_len():
    df = pd.DataFrame({"Close": [1, 2, 3]})
    sl = sig_util.precompute_sl_array(df)
    tp = sig_util.precompute_tp_array(df)
    assert len(sl) == len(df)
    assert len(tp) == len(df)
