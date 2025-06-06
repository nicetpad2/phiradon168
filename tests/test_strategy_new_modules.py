import pandas as pd
import matplotlib
matplotlib.use('Agg')
import pytest

from strategy.order_management import create_order
from strategy.risk_management import calculate_position_size
from strategy.stoploss_utils import atr_stop_loss
from strategy.trade_executor import execute_order
from strategy.plots import plot_equity_curve


def test_calculate_position_size_valid():
    lot = calculate_position_size(1000, 0.01, 10)
    assert lot >= 0.01


def test_calculate_position_size_invalid():
    with pytest.raises(ValueError):
        calculate_position_size(0, 0.01, 10)


def test_atr_stop_loss_length():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    sl = atr_stop_loss(s, period=5)
    assert len(sl) == len(s)


def test_create_and_execute_order():
    order = create_order('BUY', 1.0, 0.9, 1.1)
    pnl = execute_order(order, 1.2)
    assert pnl > 0


def test_plot_equity_curve(tmp_path):
    df = pd.DataFrame({'Equity': [1, 2, 3]})
    out_file = plot_equity_curve(df, tmp_path)
    assert out_file.exists()
