"""Stop-loss calculation helpers."""
from __future__ import annotations

import pandas as pd


def atr_stop_loss(close: pd.Series, period: int = 14) -> pd.Series:
    """Return a naive ATR-based stop loss series."""
    if len(close) < period:
        raise ValueError("close length must be >= period")
    return close.diff().abs().rolling(period).mean().fillna(method="bfill")

__all__ = ["atr_stop_loss"]
