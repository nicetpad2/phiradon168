import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from strategy import (
    drift_observer,
    entry_rules,
    exit_rules,
    metrics,
    order_management,
    plots,
    risk_management,
    stoploss_utils,
    strategy as strategy_module,
    trade_executor,
    trend_filter,
)
from src.utils.settings import Settings

def test_drift_observer_init():
    obs = drift_observer.DriftObserver(["a", "b"])
    assert obs.features == ["a", "b"] and obs.results == {}


def test_drift_observer_init_error():
    with pytest.raises(ValueError):
        drift_observer.DriftObserver("x")


def test_generate_open_signals_all(monkeypatch):
    df = pd.DataFrame({"Close": [1, 2, 3, 4], "Volume": [100, 120, 130, 150]})
    monkeypatch.setattr(entry_rules, "macd", lambda x: (None, None, pd.Series([0.1] * len(x))))
    monkeypatch.setattr(entry_rules, "rsi", lambda x: pd.Series([60] * len(x)))
    monkeypatch.setattr(entry_rules, "detect_macd_divergence", lambda a, b: "bull")
    monkeypatch.setattr(entry_rules, "sma", lambda s, p: pd.Series([p] * len(s)))
    res = entry_rules.generate_open_signals(df, trend="UP")
    assert res.dtype == np.int8 and res.sum() >= 1


def test_generate_open_signals_trend_down():
    df = pd.DataFrame({
        "Close": [1, 1.1, 1.2],
        "MACD_hist": [0.1, 0.1, 0.1],
        "RSI": [60, 60, 60],
        "MA_fast": [1, 1, 1],
        "MA_slow": [0, 0, 0],
    })
    res = entry_rules.generate_open_signals(df, use_macd=False, use_rsi=False, trend="DOWN")
    assert res.tolist() == [0, 0, 0]


def test_generate_close_signals_all(monkeypatch):
    df = pd.DataFrame({"Close": [5, 4, 3, 2]})
    monkeypatch.setattr(exit_rules, "macd", lambda x: (None, None, pd.Series([-0.1] * len(x))))
    monkeypatch.setattr(exit_rules, "rsi", lambda x: pd.Series([40] * len(x)))
    res = exit_rules.generate_close_signals(df)
    assert res.dtype == np.int8 and res.sum() >= 1


def test_precompute_arrays(monkeypatch):
    df = pd.DataFrame({"High": [1, 2], "Low": [0.5, 1.5], "Close": [1, 2]})
    monkeypatch.setattr(exit_rules, "atr", lambda d, p: d.assign(ATR_14=[0.1, 0.2]))
    sl = exit_rules.precompute_sl_array(df)
    tp = exit_rules.precompute_tp_array(df)
    assert sl.tolist() == pytest.approx([0.15, 0.3])
    assert tp.tolist() == pytest.approx([0.25, 0.5])


def test_calculate_metrics_edge_cases():
    assert metrics.calculate_metrics([]) == {"r_multiple": 0.0, "winrate": 0.0}
    res = metrics.calculate_metrics([1.0, -0.5, 0.5])
    assert res["r_multiple"] == 1.0 and res["winrate"] == pytest.approx(2 / 3)


def test_order_manager_flow():
    settings = Settings(cooldown_secs=1, kill_switch_pct=0.5)
    om = order_management.OrderManager(settings=settings)
    now = datetime.now(timezone.utc)
    assert om.place_order({}, now) is order_management.OrderStatus.OPEN
    assert om.place_order({}, now + timedelta(seconds=0.5)) is order_management.OrderStatus.BLOCKED_COOLDOWN
    om.update_drawdown(0.6)
    assert om.place_order({}, now + timedelta(seconds=2)) is order_management.OrderStatus.KILL_SWITCH


def test_order_helpers():
    order = order_management.place_order("BUY", 1.0, 0.9, 1.1, 0.01)
    assert order["side"] == "BUY" and order["tp_price"] == 1.1
    legacy = order_management.create_order("SELL", 1.0, 0.8, 1.2)
    assert legacy["sl"] == 0.8
    with pytest.raises(ValueError):
        order_management.create_order("BUY", 1.0, None, 1.1)
    with pytest.raises(ValueError):
        order_management.create_order("HOLD", 1.0, 0.9, 1.1)


def test_plot_equity_curve(tmp_path):
    ax = plots.plot_equity_curve([1, 2, 3])
    assert hasattr(ax, "plot")
    out = plots.plot_equity_curve(pd.DataFrame({"Equity": [1, 2, 3]}), tmp_path)
    assert out.exists()


def test_risk_management_utils():
    assert risk_management.dynamic_position_size(1.0, 2.0, 1.0) < 1.0
    assert risk_management.dynamic_position_size(1.0, 0.5, 1.0) == 0.05
    assert risk_management.dynamic_position_size(1.0, 1.0, 1.0) == 1.0
    with pytest.raises(ValueError):
        risk_management.compute_lot_size(0, 0.01, 10)
    assert risk_management.check_max_daily_drawdown(100.0, 98.0)
    assert risk_management.check_trailing_equity_stop(110.0, 100.0)
    assert risk_management.can_open_trade(1, 2)
    assert risk_management.should_hard_cutoff(0.05, 6)
    rm = risk_management.RiskManager(settings=Settings(kill_switch_pct=0.3))
    rm.update_drawdown(0.4)
    assert rm.check_kill_switch() is risk_management.OrderStatus.KILL_SWITCH


def test_stoploss_utils():
    close = pd.Series(range(20))
    sl = stoploss_utils.atr_stop_loss(close, period=5)
    assert len(sl) == len(close)
    buy_sl, buy_tp = stoploss_utils.atr_sl_tp_wrapper(1.0, 0.1, "BUY")
    sell_sl, sell_tp = stoploss_utils.atr_sl_tp_wrapper(1.0, 0.1, "SELL")
    assert buy_sl < 1.0 < buy_tp
    assert sell_tp < 1.0 < sell_sl
    with pytest.raises(ValueError):
        stoploss_utils.atr_stop_loss(pd.Series([1, 2, 3]), period=5)


def test_trade_executor():
    trade = trade_executor.open_trade("BUY", 1.0, 0.9, 1.1, 0.01)
    assert trade_executor.execute_order({"side": "BUY", "entry_price": 1.0, "sl": 0.9, "tp": 1.1}, 1.2) == pytest.approx(0.2)
    assert trade_executor.execute_order({"side": "SELL", "entry_price": 1.0, "sl": 1.1, "tp": 0.9}, 0.8) == pytest.approx(0.2)
    with pytest.raises(KeyError):
        trade_executor.execute_order({"side": "BUY"}, 1.1)
    with pytest.raises(ValueError):
        trade_executor.execute_order({"side": "BUY", "entry_price": 1.0, "sl": 0.9, "tp": 1.1}, None)


def test_apply_trend_filter_cases():
    df = pd.DataFrame({"Trend_Zone": ["UP", "DOWN", "NEUTRAL"], "Entry_Long": [1, 1, 1], "Entry_Short": [1, 1, 1]})
    res = trend_filter.apply_trend_filter(df)
    assert res["Entry_Long"].tolist() == [1, 0, 0]
    assert res["Entry_Short"].tolist() == [0, 1, 0]
