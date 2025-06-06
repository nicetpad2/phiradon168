import json
import os
import sys
import logging

import pytest
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))

from src.adaptive import (
    adaptive_sl_tp,
    adaptive_risk,
    log_best_params,
    compute_kelly_position,
    compute_dynamic_lot,
    calculate_atr,
    atr_position_size,
    compute_trailing_atr_stop,
    volatility_adjusted_lot_size,
    dynamic_risk_adjustment,
    check_portfolio_stop,
    calculate_dynamic_sl_tp,
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


def test_compute_kelly_position_valid():
    val = compute_kelly_position(0.6, 2)
    assert 39.9 < val < 40.1


def test_compute_kelly_position_invalid_ratio(caplog):
    with caplog.at_level(logging.WARNING):
        val = compute_kelly_position(0.6, 0)
    assert val == 0.0
    assert any("win_loss_ratio" in m for m in caplog.messages)


def test_compute_dynamic_lot_reductions():
    assert compute_dynamic_lot(1.0, 0.11) == 0.5
    assert compute_dynamic_lot(1.0, 0.07) == 0.75
    assert compute_dynamic_lot(1.0, 0.02) == 1.0
    assert compute_dynamic_lot(1.0, "x") == 1.0


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


def test_compute_trailing_atr_stop_buy():
    new_sl = compute_trailing_atr_stop(10.0, 11.2, 1.0, 'BUY', 9.5)
    assert new_sl == 10.0
    new_sl2 = compute_trailing_atr_stop(10.0, 12.5, 1.0, 'BUY', 10.0)
    assert new_sl2 > 10.0


def test_compute_trailing_atr_stop_sell():
    new_sl = compute_trailing_atr_stop(10.0, 8.8, 1.0, 'SELL', 10.5)
    assert new_sl == 10.0
    new_sl2 = compute_trailing_atr_stop(10.0, 7.5, 1.0, 'SELL', 10.0)
    assert new_sl2 < 10.0


def test_trailing_stop_case_insensitive():
    assert compute_trailing_atr_stop(1.0, 2.0, 0.5, 'buy', 0.5) > 0.5
    assert compute_trailing_atr_stop(1.0, 0.5, 0.5, 'Sell', 1.5) < 1.5


def test_volatility_adjusted_lot_size():
    lot, sl = volatility_adjusted_lot_size(1000, 0.2, sl_multiplier=2.0,
                                           pip_value=0.1, risk_pct=0.01)
    assert lot > 0
    assert sl == 0.4


def test_dynamic_risk_adjustment():
    risk = dynamic_risk_adjustment([-0.06, -0.07, -0.06], base_risk=0.01)
    assert risk == 0.005
    risk2 = dynamic_risk_adjustment([0.06, 0.05], base_risk=0.01)
    assert risk2 == 0.015
    risk3 = dynamic_risk_adjustment([], base_risk=0.01)
    assert risk3 == 0.01


def test_check_portfolio_stop():
    assert check_portfolio_stop(0.12)
    assert not check_portfolio_stop(0.05)


def test_calculate_dynamic_sl_tp_cases():
    sl, tp = calculate_dynamic_sl_tp(2.0, 0.35)
    assert sl == 3.0 and tp == 9.0

    sl2, tp2 = calculate_dynamic_sl_tp(0.5, 0.55)
    assert sl2 == 2.0 and tp2 == 3.0

    sl3, tp3 = calculate_dynamic_sl_tp(1.5, 0.45)
    assert abs(sl3 - 2.25) < 1e-9
    assert abs(tp3 - 4.5) < 1e-9
import math
import pandas as pd


def test_calculate_atr_invalid(monkeypatch):
    assert math.isnan(calculate_atr(123))
    df = pd.DataFrame({'High': [1], 'Low': [0.5], 'Close': [0.8]})
    monkeypatch.setattr(features, 'atr', lambda df_in, period=14: pd.DataFrame({'OTHER': [0.1]}))
    assert math.isnan(calculate_atr(df))


@pytest.mark.parametrize('equity,atr', [
    ('x', 1.0),
    (1.0, 'y'),
])
def test_atr_position_size_invalid_types(equity, atr):
    lot, sl = atr_position_size(equity, atr)
    assert lot == 0.01 and math.isnan(sl)


def test_atr_position_size_negative_and_small():
    lot1, sl1 = atr_position_size(0, 1)
    assert lot1 == 0.01 and math.isnan(sl1)
    lot2, sl2 = atr_position_size(1000, 1, pip_value=1e-11)
    assert lot2 == 0.01 and abs(sl2 - 1.5) < 1e-9


def test_compute_kelly_position_invalid_inputs(caplog):
    with caplog.at_level(logging.WARNING):
        val = compute_kelly_position('a', 'b')
    assert val == 0.0
    assert any('Invalid Kelly inputs' in m for m in caplog.messages)


def test_trailing_atr_stop_invalid_and_unknown(caplog):
    with caplog.at_level(logging.WARNING):
        assert compute_trailing_atr_stop('x', 1.0, 0.5, 'BUY', 0.5) == 0.5
    with caplog.at_level(logging.WARNING):
        assert compute_trailing_atr_stop(1.0, 1.0, -0.1, 'BUY', 0.5) == 0.5
    assert compute_trailing_atr_stop(1.0, 1.1, 0.5, 'HOLD', 0.5) == 0.5


@pytest.mark.parametrize('equity,atr', [
    ('x', 0.2),
    (1000, 'x'),
])
def test_volatility_adjusted_lot_size_invalid_inputs(equity, atr):
    lot, sl = volatility_adjusted_lot_size(equity, atr)
    assert lot == 0.01 and math.isnan(sl)


def test_volatility_adjusted_lot_size_edge_cases():
    lot1, sl1 = volatility_adjusted_lot_size(0, 1)
    assert lot1 == 0.01 and math.isnan(sl1)
    lot2, sl2 = volatility_adjusted_lot_size(1000, 1, pip_value=1e-11)
    assert lot2 == 0.01 and abs(sl2 - 1.5) < 1e-9


def test_dynamic_risk_adjustment_base_case():
    risk = dynamic_risk_adjustment([0.01, -0.01], base_risk=0.02)
    assert risk == 0.02


def test_check_portfolio_stop_invalid(caplog):
    with caplog.at_level(logging.WARNING):
        assert not check_portfolio_stop('x')


def test_calculate_dynamic_sl_tp_invalid():
    sl, tp = calculate_dynamic_sl_tp('x', 'y')
    assert math.isnan(sl) and math.isnan(tp)
