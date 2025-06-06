"""Stop-loss calculation helpers."""
from __future__ import annotations

import pandas as pd


def atr_stop_loss(close: pd.Series, period: int = 14) -> pd.Series:
    """Return a naive ATR-based stop loss series."""
    if len(close) < period:
        raise ValueError("close length must be >= period")
    return close.diff().abs().rolling(period).mean().bfill()

__all__ = ["atr_stop_loss"]

def atr_sl_tp_wrapper(price: float, atr: float, side: str) -> tuple[float, float]:
    """Return basic SL/TP pair based on ATR distance."""
    if side == "BUY":
        return price - atr, price + atr
    else:
        return price + atr, price - atr

__all__.append("atr_sl_tp_wrapper")
