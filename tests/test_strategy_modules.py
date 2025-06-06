import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from strategy import (
    apply_strategy,
    create_order,
    calculate_position_size,
    atr_stop_loss,
    execute_order,
    plot_equity_curve,
)


def test_create_order_contains_sl_tp():
    order = create_order("BUY", 1.0, 0.9, 1.1)
    assert order["sl"] == 0.9
    assert order["tp"] == 1.1


def test_calculate_position_size_basic():
    size = calculate_position_size(1000, 0.01, 10)
    assert size == 1.0


def test_atr_stop_loss():
    s = pd.Series(range(20))
    sl = atr_stop_loss(s, period=5)
    assert len(sl) == len(s)


def test_apply_strategy_simple():
    df = pd.DataFrame({"Close": [1.0, 1.1, 1.0]})
    result = apply_strategy(df)
    assert "Entry" in result and "Exit" in result


def test_create_and_execute_order():
    order = create_order("BUY", 1.0, 0.9, 1.1)
    pnl = execute_order(order, 1.2)
    assert pnl > 0


def test_plot_equity_curve_returns_fig():
    fig = plot_equity_curve([1, 2, 3])
    assert hasattr(fig, "savefig")
