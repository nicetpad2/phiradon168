import os
import sys
import types
from datetime import datetime, timezone
import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import src.order_manager as om


def _setup_strategy(monkeypatch, **attrs):
    stub = types.SimpleNamespace(**attrs)
    monkeypatch.setitem(sys.modules, 'src.strategy', stub)
    import src
    monkeypatch.setattr(src, 'strategy', stub, raising=False)
    return stub


def test_check_main_exit_conditions(monkeypatch):
    _setup_strategy(monkeypatch, MAX_HOLDING_BARS=2)
    base_order = {
        'side': 'BUY',
        'sl_price': 9.8,
        'tp_price': 10.2,
        'entry_price': 10.0,
        'entry_bar_count': 0,
        'entry_time': 't'
    }
    now = pd.Timestamp('2023-01-01')
    order = base_order.copy()
    order.update({'be_triggered': True, 'sl_price': 10.0})
    row = pd.Series({'High': 10.1, 'Low': 9.9, 'Close': 10.0})
    closed, price, reason, _ = om.check_main_exit_conditions(order, row, 1, now)
    assert closed and reason == 'BE-SL' and price == 10.0
    order = base_order.copy()
    row = pd.Series({'High': 10.0, 'Low': 9.6, 'Close': 9.7})
    closed, price, reason, _ = om.check_main_exit_conditions(order, row, 1, now)
    assert closed and reason == 'SL' and price == 9.8
    order = base_order.copy()
    row = pd.Series({'High': 10.3, 'Low': 9.9, 'Close': 10.2})
    closed, price, reason, _ = om.check_main_exit_conditions(order, row, 1, now)
    assert closed and reason == 'TP' and price == 10.2
    order = base_order.copy()
    row = pd.Series({'High': 10.0, 'Low': 9.9, 'Close': 9.9})
    closed, price, reason, _ = om.check_main_exit_conditions(order, row, 2, now)
    assert closed and reason.startswith('MaxBars') and price == 9.9
    order = base_order.copy()
    row = pd.Series({'High': 10.0, 'Low': 9.9, 'Close': float('nan')})
    closed, price, reason, _ = om.check_main_exit_conditions(order, row, 2, now)
    assert closed and reason.endswith('CloseNaN') and price == 9.8


def test_update_open_order_state_be(monkeypatch):
    _setup_strategy(
        monkeypatch,
        DYNAMIC_BE_ATR_THRESHOLD_HIGH=0.0,
        DYNAMIC_BE_R_ADJUST_HIGH=1.0,
        ADAPTIVE_TSL_START_ATR_MULT=1.0,
        update_breakeven_half_tp=lambda order, h, l, now: (order, True),
        update_tsl_only=lambda *a, **k: (a[0], False),
        compute_trailing_atr_stop=lambda *a, **k: a[4],
        update_trailing_tp2=lambda o, *a: o,
        dynamic_tp2_multiplier=lambda *a, **k: 2.0,
    )
    order = {
        'side': 'BUY',
        'entry_price': 10.0,
        'original_sl_price': 9.0,
        'sl_price': 9.0,
        'atr_at_entry': 2.0,
        'entry_time': 't'
    }
    now = datetime.now(timezone.utc)
    order, be, tsl, be_c, tsl_c = om.update_open_order_state(
        order,
        current_high=12.0,
        current_low=9.5,
        current_atr=2.0,
        avg_atr=1.0,
        now=now,
        base_be_r_thresh=1.0,
        fold_sl_multiplier_base=2.0,
        base_tp_multiplier_config=2.0,
        be_sl_counter=0,
        tsl_counter=0,
    )
    assert be and order['be_triggered'] and not tsl and be_c == 2


def test_update_open_order_state_tsl(monkeypatch):
    _setup_strategy(
        monkeypatch,
        DYNAMIC_BE_ATR_THRESHOLD_HIGH=1e9,
        DYNAMIC_BE_R_ADJUST_HIGH=0.0,
        ADAPTIVE_TSL_START_ATR_MULT=1.0,
        update_breakeven_half_tp=lambda order, h, l, now: (order, False),
        update_tsl_only=lambda order, *a, **k: (order, True),
        compute_trailing_atr_stop=lambda *a, **k: 10.4,
        update_trailing_tp2=lambda o, *a: o,
        dynamic_tp2_multiplier=lambda *a, **k: 1.5,
    )
    order = {
        'side': 'SELL',
        'entry_price': 10.0,
        'original_sl_price': 11.0,
        'sl_price': 11.0,
        'atr_at_entry': 1.0,
        'entry_time': 't'
    }
    now = datetime.now(timezone.utc)
    order, be, tsl, be_c, tsl_c = om.update_open_order_state(
        order,
        current_high=10.0,
        current_low=8.0,
        current_atr=1.0,
        avg_atr=1.0,
        now=now,
        base_be_r_thresh=0.0,
        fold_sl_multiplier_base=2.0,
        base_tp_multiplier_config=2.0,
        be_sl_counter=0,
        tsl_counter=0,
    )
    assert not be and tsl and order['sl_price'] == 10.4 and tsl_c == 1
