import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from src.adaptive import (
    adaptive_sl_tp,
    adaptive_risk,
    log_best_params,
    calculate_atr,
    atr_position_size,
)
import src.features as features


def test_adaptive_sl_tp_high_vol():
    sl, tp = adaptive_sl_tp(2.0, 1.0, base_sl=2.0, base_tp=1.8)
    assert sl > 2.0 and tp > 1.8


def test_adaptive_sl_tp_invalid_and_low_vol():
    # invalid input triggers fallback
    assert adaptive_sl_tp('x', 'y') == (2.0, 1.8)
    # low volatility adjusts downwards
    sl, tp = adaptive_sl_tp(0.5, 1.0)
    assert sl < 2.0 and tp < 1.8
    # zero ATR average falls back to base values
    assert adaptive_sl_tp(1.0, 0.0) == (2.0, 1.8)
    # mid ratio returns base values
    assert adaptive_sl_tp(1.0, 1.0) == (2.0, 1.8)


def test_adaptive_risk_reduce():
    risk = adaptive_risk(80, 100, base_risk=0.01, dd_threshold=0.1)
    assert risk < 0.01


def test_adaptive_risk_edge_cases():
    assert adaptive_risk('x', 100) == 0.01
    assert adaptive_risk(100, 0) == 0.01
    risk = adaptive_risk(50, 100, base_risk=0.01, dd_threshold=0.1)
    assert risk < 0.01
    assert adaptive_risk(120, 100, base_risk=0.01, dd_threshold=0.1) == 0.01


def test_log_best_params(tmp_path):
    path = log_best_params({"a": 1}, 0, tmp_path)
    assert path is not None and os.path.isfile(path)
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    assert data["a"] == 1


def test_calculate_atr_and_position_size(monkeypatch):
    import pandas as pd

    df = pd.DataFrame({"High": [1, 2], "Low": [0.5, 1.5], "Close": [0.8, 1.8]})

    monkeypatch.setattr(
        features,
        "atr",
        lambda df_in, period=14: pd.DataFrame({"ATR_14": [0.2, 0.3]}, index=df_in.index),
    )

    atr_val = calculate_atr(df, period=14)
    assert atr_val == 0.3

    lot, sl = atr_position_size(1000, atr_val, risk_pct=0.01, atr_mult=1.5, pip_value=0.1)
    assert lot > 0.0 and sl == atr_val * 1.5
