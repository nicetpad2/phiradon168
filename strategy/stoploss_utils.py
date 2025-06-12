"""Stop-loss calculation helpers."""
from __future__ import annotations

import pandas as pd
from src.adaptive import compute_trailing_atr_stop


def atr_stop_loss(close: pd.Series, period: int = 14) -> pd.Series:
    """Return a naive ATR-based stop loss distance series."""
    if len(close) < period:
        raise ValueError("close length must be >= period")
    return close.diff().abs().rolling(period).mean().bfill()

__all__ = ["atr_stop_loss", "atr_trailing_stop"]


def atr_trailing_stop(
    entry_price: float,
    current_price: float,
    atr: float,
    side: str,
    current_sl: float,
    atr_mult: float = 1.5,
) -> float:
    """[Patch v6.8.5] Return updated stop loss using ATR trailing logic."""
    return compute_trailing_atr_stop(
        entry_price, current_price, atr, side, current_sl, atr_mult
    )

def atr_sl_tp_wrapper(price: float, atr: float, side: str) -> tuple[float, float]:
    """Return basic SL/TP pair based on ATR distance."""
    if side == "BUY":
        return price - atr, price + atr
    else:
        return price + atr, price - atr

__all__.append("atr_sl_tp_wrapper")
__all__.append("atr_trailing_stop")
