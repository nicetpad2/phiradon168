import os
import sys
from datetime import datetime, timezone
import types

import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import src.order_manager as om

from tests.test_order_manager_module import _setup_strategy


@pytest.mark.parametrize(
    "row, reason, price",
    [
        (pd.Series({'High': 10.2, 'Low': 7.9, 'Close': 9.8}), 'BE-SL', 10.0),
        (pd.Series({'High': 11.2, 'Low': 9.8, 'Close': 10.5}), 'SL', 11.0),
        (pd.Series({'High': 9.8, 'Low': 8.0, 'Close': 8.6}), 'TP', 8.6),
    ],
)
def test_check_main_exit_conditions_sell(monkeypatch, row, reason, price):
    _setup_strategy(monkeypatch, MAX_HOLDING_BARS=0)
    order = {
        'side': 'SELL',
        'sl_price': 11.0,
        'tp_price': 8.6,
        'entry_price': 10.0,
        'entry_bar_count': 0,
        'entry_time': 't',
    }
    if reason == 'BE-SL':
        order['be_triggered'] = True
        order['sl_price'] = 10.0
    now = pd.Timestamp('2023-01-01')
    closed, exit_price, exit_reason, _ = om.check_main_exit_conditions(order, row, 1, now)
    assert closed and exit_reason.startswith(reason) and exit_price == price


def test_check_main_exit_conditions_zero_fallback(monkeypatch):
    _setup_strategy(monkeypatch, MAX_HOLDING_BARS=0)
    order = {
        'side': 'BUY',
        'sl_price': float('nan'),
        'tp_price': float('nan'),
        'entry_price': float('nan'),
        'entry_bar_count': 0,
        'entry_time': 't',
    }
    row = pd.Series({'High': 1.0, 'Low': 0.8, 'Close': float('nan')})
    closed, price, reason, _ = om.check_main_exit_conditions(order, row, 1, pd.Timestamp('2023-01-01'))
    assert closed and reason.endswith('CloseNaN') and price == 0


def test_update_open_order_state_dynamic_be_sell(monkeypatch):
    _setup_strategy(
        monkeypatch,
        DYNAMIC_BE_ATR_THRESHOLD_HIGH=0.0,
        DYNAMIC_BE_R_ADJUST_HIGH=1.0,
        ADAPTIVE_TSL_START_ATR_MULT=1.0,
        update_breakeven_half_tp=lambda o, h, l, now: (o, False),
        update_tsl_only=lambda *a, **k: (a[0], False),
        compute_trailing_atr_stop=lambda *a, **k: a[4],
        update_trailing_tp2=lambda o, a, b: o,
        dynamic_tp2_multiplier=lambda *a, **k: 1.5,
    )
    order = {
        'side': 'SELL',
        'entry_price': 10.0,
        'original_sl_price': 11.0,
        'sl_price': 11.0,
        'atr_at_entry': 2.0,
        'entry_time': 't',
    }
    now = datetime.now(timezone.utc)
    order, be, tsl, be_c, _ = om.update_open_order_state(
        order,
        current_high=10.0,
        current_low=7.9,
        current_atr=2.0,
        avg_atr=1.0,
        now=now,
        base_be_r_thresh=1.0,
        fold_sl_multiplier_base=2.0,
        base_tp_multiplier_config=2.0,
        be_sl_counter=0,
        tsl_counter=0,
    )
    assert be and order['be_triggered'] and be_c == 1 and order['sl_price'] == 10.0


def test_update_open_order_state_tsl_buy(monkeypatch):
    _setup_strategy(
        monkeypatch,
        DYNAMIC_BE_ATR_THRESHOLD_HIGH=1e9,
        DYNAMIC_BE_R_ADJUST_HIGH=0.0,
        ADAPTIVE_TSL_START_ATR_MULT=1.0,
        update_breakeven_half_tp=lambda o, h, l, now: (o, False),
        update_tsl_only=lambda order, *a, **k: (order, True),
        compute_trailing_atr_stop=lambda *a, **k: 9.4,
        update_trailing_tp2=lambda o, a, b: o,
        dynamic_tp2_multiplier=lambda *a, **k: 2.0,
    )
    order = {
        'side': 'BUY',
        'entry_price': 10.0,
        'original_sl_price': 9.0,
        'sl_price': 9.0,
        'atr_at_entry': 1.0,
        'entry_time': 't',
    }
    now = datetime.now(timezone.utc)
    order, be, tsl, _, tsl_c = om.update_open_order_state(
        order,
        current_high=11.5,
        current_low=9.5,
        current_atr=1.0,
        avg_atr=1.0,
        now=now,
        base_be_r_thresh=0.0,
        fold_sl_multiplier_base=2.0,
        base_tp_multiplier_config=2.0,
        be_sl_counter=0,
        tsl_counter=0,
    )
    assert not be and tsl and order['peak_since_tsl_activation'] == 11.5 and tsl_c == 1


def test_update_open_order_state_error_logging(monkeypatch, caplog):
    orig_to_numeric = om.pd.to_numeric

    def broken_numeric(val, *a, **k):
        if val == 'bad':
            raise ValueError('boom')
        return orig_to_numeric(val, *a, **k)

    monkeypatch.setattr(om.pd, 'to_numeric', broken_numeric)
    _setup_strategy(
        monkeypatch,
        DYNAMIC_BE_ATR_THRESHOLD_HIGH=0.0,
        DYNAMIC_BE_R_ADJUST_HIGH=1.0,
        ADAPTIVE_TSL_START_ATR_MULT=1.0,
        update_breakeven_half_tp=lambda o, h, l, now: (o, False),
        update_tsl_only=lambda *a, **k: (a[0], False),
        compute_trailing_atr_stop=lambda *a, **k: a[4],
        update_trailing_tp2=lambda o, a, b: o,
        dynamic_tp2_multiplier=lambda *a, **k: 1.5,
    )
    caplog.set_level(logging.WARNING)
    order = {
        'side': 'BUY',
        'entry_price': 10.0,
        'original_sl_price': 9.0,
        'sl_price': 9.0,
        'atr_at_entry': 1.0,
        'entry_time': 't',
    }
    now = datetime.now(timezone.utc)
    om.update_open_order_state(
        order,
        current_high=10.5,
        current_low=9.5,
        current_atr=1.0,
        avg_atr='bad',
        now=now,
        base_be_r_thresh=1.0,
        fold_sl_multiplier_base=2.0,
        base_tp_multiplier_config=2.0,
        be_sl_counter=0,
        tsl_counter=0,
    )
    assert any('Error calculating dynamic BE threshold' in r.message for r in caplog.records)
