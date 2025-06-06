import os
import sys
import pandas as pd
from datetime import datetime, timezone

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from strategy import (
    run_backtest,
    OrderManager,
    calculate_position_size,
    atr_sl_tp_wrapper,
    open_trade,
    plot_equity_curve,
)


def test_order_manager_place_order():
    om = OrderManager()
    now = datetime.now(timezone.utc)
    status = om.place_order({}, now)
    assert status.name == "OPEN"


def test_calculate_position_size_basic():
    size = calculate_position_size(1000, 0.01, 10)
    assert size == 1.0


def test_atr_sl_tp_wrapper():
    sl, tp = atr_sl_tp_wrapper(1.0, 0.1, "BUY")
    assert sl < 1.0 and tp > 1.0


def test_run_backtest_simple():
    df = pd.DataFrame({"Close": [1.0, 1.1, 1.0]})
    orders = run_backtest(df, 1000)
    assert isinstance(orders, list)


def test_plot_equity_curve_returns_axis():
    ax = plot_equity_curve([1, 2, 3])
    assert hasattr(ax, "plot")

