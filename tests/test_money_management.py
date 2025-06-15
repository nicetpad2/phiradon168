import pytest
from src.money_management import (
    atr_sl_tp,
    update_be_trailing,
    adaptive_position_size,
    portfolio_hard_stop,
)


def test_atr_sl_tp_buy_sell():
    sl, tp = atr_sl_tp(10.0, 0.5, 'BUY')
    assert sl == 9.0 and tp == 11.0
    sl2, tp2 = atr_sl_tp(10.0, 0.5, 'SELL')
    assert sl2 == 11.0 and tp2 == 9.0

    sl3, tp3 = atr_sl_tp(10.0, 0.5, 'BUY', sl_mult=1.0, tp_mult=3.0)
    assert sl3 == 9.5 and tp3 == 11.5


def test_update_be_trailing():
    order = {'entry_price': 10.0, 'sl_price': 9.5, 'be_triggered': False}
    order = update_be_trailing(order, 10.5, 0.5, 'BUY')
    assert order['be_triggered'] and order['sl_price'] == 10.0
    order = update_be_trailing(order, 11.0, 0.5, 'BUY')
    assert order['sl_price'] > 10.0


def test_adaptive_position_size():
    lot = adaptive_position_size(1000.0, 0.2)
    assert lot > 0.0


def test_portfolio_hard_stop():
    assert portfolio_hard_stop(1000.0, 890.0)
    assert not portfolio_hard_stop(1000.0, 950.0)
