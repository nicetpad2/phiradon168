"""High-level strategy orchestration utilities."""
from __future__ import annotations

from typing import List, Dict
import pandas as pd

from .entry_rules import generate_open_signals
from .exit_rules import generate_close_signals
from .order_management import place_order
from .risk_management import calculate_position_size


# [Patch v6.9.3] Restore apply_strategy and run_backtest helpers

def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with entry and exit signals."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    result = df.copy()
    result["Entry"] = generate_open_signals(df)
    result["Exit"] = generate_close_signals(df)
    return result

__all__ = ["apply_strategy"]


def run_backtest(df: pd.DataFrame, balance: float, risk_pct: float = 0.01) -> List[Dict]:
    """Run a simple backtest over the given DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    orders = []
    position = None
    for idx, row in df.iterrows():
        if position is None:
            open_sig = generate_open_signals(df.loc[[idx]])
            if open_sig[0] == 1:
                size = calculate_position_size(balance, risk_pct, row['Close'] * 0.01)
                sl = row['Close'] - row['Close'] * 0.01
                tp = row['Close'] + row['Close'] * 0.02
                position = place_order("BUY", row['Close'], sl, tp, size)
        else:
            close_sig = generate_close_signals(df.loc[[idx]])
            if close_sig[0] == 1:
                position['exit_price'] = row['Close']
                orders.append(position)
                position = None
    if position is not None:
        position['exit_price'] = df.iloc[-1]['Close']
        orders.append(position)
    return orders

__all__.append("run_backtest")
