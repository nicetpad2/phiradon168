import os
import sys
import math
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.money_management import (
    atr_sl_tp,
    update_be_trailing,
    adaptive_position_size,
    portfolio_hard_stop,
)


def test_atr_sl_tp_invalid_inputs():
    sl, tp = atr_sl_tp('a', 'b', 'BUY')
    assert math.isnan(sl) and math.isnan(tp)
    sl, tp = atr_sl_tp(10.0, -1.0, 'BUY')
    assert math.isnan(sl) and math.isnan(tp)
    sl, tp = atr_sl_tp(10.0, 0.5, 'HOLD')
    assert math.isnan(sl) and math.isnan(tp)


def test_update_be_trailing_edge_cases(monkeypatch):
    assert update_be_trailing(None, 1.0, 1.0, 'BUY') == {}

    bad_order = {'entry_price': 'a', 'sl_price': 1.0}
    assert update_be_trailing(bad_order, 1.0, 1.0, 'BUY') is bad_order

    calls = {}
    def fake_trailing(entry, price, atr, side, sl, mult):
        calls['args'] = (entry, price, atr, side, sl, mult)
        return price + 0.1

    order = {'entry_price': 10.0, 'sl_price': 10.5, 'be_triggered': False}
    order = update_be_trailing(order, 9.0, 1.0, 'SELL')
    assert order['be_triggered'] and order['sl_price'] == 10.0

    monkeypatch.setattr('src.money_management.compute_trailing_atr_stop', fake_trailing)
    order = update_be_trailing(order, 8.0, 1.0, 'SELL')
    assert order['sl_price'] == min(10.0, 8.1)
    assert calls['args'][3] == 'SELL'


def test_adaptive_position_size_and_portfolio_stop(monkeypatch):
    monkeypatch.setattr('src.money_management.volatility_adjusted_lot_size', lambda e, a, risk_pct=0.01: (0.5, 1.0))
    assert adaptive_position_size(1000.0, 0.2) == 0.5

    monkeypatch.setattr('src.money_management.check_portfolio_stop', lambda dd, t: dd >= t)
    assert portfolio_hard_stop(1000.0, 800.0)
    assert not portfolio_hard_stop('bad', 800.0)
    assert not portfolio_hard_stop(-1, 0)
