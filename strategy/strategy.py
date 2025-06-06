"""Simple high-level strategy orchestration utilities."""
from __future__ import annotations

import pandas as pd
from .entry_rules import generate_open_signals
from .exit_rules import generate_close_signals


def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with entry and exit signals."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    result = df.copy()
    result["Entry"] = generate_open_signals(df)
    result["Exit"] = generate_close_signals(df)
    return result

__all__ = ["apply_strategy"]

def run_backtest(df: pd.DataFrame, initial_balance: float) -> list:
    """Dummy backtest returning an empty list."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    return []

__all__.append("run_backtest")
