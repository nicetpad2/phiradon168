"""Core strategy loop using entry and exit rules."""

from typing import List, Dict

import pandas as pd

from .entry_rules import generate_open_signals
from .exit_rules import generate_close_signals
from .order_management import place_order
from .risk_management import calculate_position_size

__all__ = ["run_backtest"]


def run_backtest(df: pd.DataFrame, balance: float, risk_pct: float = 0.01) -> List[Dict]:
    """Run a simple backtest over the given DataFrame."""
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
