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
    compute_kelly_position,
    compute_dynamic_lot,
)


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


def test_compute_kelly_position_valid():
    val = compute_kelly_position(0.6, 2)
    assert 0.39 < val < 0.41


def test_compute_dynamic_lot_reductions():
    assert compute_dynamic_lot(1.0, 0.11) == 0.5
    assert compute_dynamic_lot(1.0, 0.07) == 0.75
    assert compute_dynamic_lot(1.0, 0.02) == 1.0
    assert compute_dynamic_lot(1.0, "x") == 1.0
