"""High-level strategy helpers and minimal backtest routine."""
from __future__ import annotations

import pandas as pd

from .entry_rules import generate_open_signals
from .exit_rules import generate_close_signals
from .order_management import OrderManager, OrderStatus as OrderStatusOM
from .risk_management import (
    calculate_position_size,
    RiskManager,
    OrderStatus as OrderStatusRM,
)
from .stoploss_utils import atr_sl_tp_wrapper
from .trade_executor import execute_order
from .order_management import create_order as legacy_order


def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with ``Entry`` and ``Exit`` signals."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    result = df.copy()
    result["Entry"] = generate_open_signals(df)
    result["Exit"] = generate_close_signals(df)
    return result


__all__ = ["apply_strategy"]


def run_backtest(df: pd.DataFrame, initial_balance: float, risk_pct: float = 0.01, allow_short: bool = True) -> list:
    """Run a minimal backtest using generated signals.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLC data with ``Close`` prices.
    initial_balance : float
        Starting account balance.
    risk_pct : float, optional
        Percentage risked per trade, by default ``0.01``.

    Returns
    -------
    list
        A list of executed trade dictionaries.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if initial_balance <= 0 or df.empty:
        return []

    data = df.copy()
    data["Entry"] = generate_open_signals(df, allow_short=allow_short)
    data["Exit"] = generate_close_signals(df)
    order_mgr = OrderManager()
    risk_mgr = RiskManager()

    balance = initial_balance
    current_trade: dict | None = None
    trades: list[dict] = []

    for idx, row in data.iterrows():
        if current_trade:
            sl_key = "sl_price" if "sl_price" in current_trade else "sl"
            tp_key = "tp_price" if "tp_price" in current_trade else "tp"
            sl_hit = (
                row["Close"] <= current_trade[sl_key]
                if current_trade.get("side") == "BUY"
                else row["Close"] >= current_trade[sl_key]
            )
            tp_hit = (
                row["Close"] >= current_trade[tp_key]
                if current_trade.get("side") == "BUY"
                else row["Close"] <= current_trade[tp_key]
            )
            if row["Exit"] == 1 or sl_hit or tp_hit:
                profit = execute_order(current_trade, row["Close"])
                balance += profit * current_trade["size"]
                trades.append(
                    {
                        "entry_idx": current_trade["entry_idx"],
                        "exit_idx": idx,
                        "profit": profit,
                    }
                )
                drawdown = max(0.0, (initial_balance - balance) / initial_balance)
                risk_mgr.update_drawdown(drawdown)
                current_trade = None
                if risk_mgr.check_kill_switch() is OrderStatusRM.KILL_SWITCH:
                    break
        else:
            if row["Entry"] in (1, -1):
                side = "BUY" if row["Entry"] == 1 else "SELL"
                status = order_mgr.place_order({}, idx)
                if status is OrderStatusOM.OPEN:
                    sl, tp = atr_sl_tp_wrapper(row["Close"], row.get("ATR_14", 1.0), side)
                    size = calculate_position_size(
                        balance, risk_pct, abs(row["Close"] - sl)
                    )
                    current_trade = legacy_order(side, row["Close"], sl, tp)
                    current_trade["size"] = size
                    current_trade["entry_idx"] = idx

    return trades


__all__.append("run_backtest")
